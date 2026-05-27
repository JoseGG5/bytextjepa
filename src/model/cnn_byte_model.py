import torch
import torch.nn.functional as F
from torch import nn
from transformers import ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import BaseModelOutput

from src.utils import mean_pooling


class CnnByteBackbone(nn.Module):
    """Compresses byte embeddings with a local CNN before the transformer."""

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        self.kernel_size = int(model_cfg["conv_kernel_size"])
        self.stride = int(model_cfg["conv_stride"])
        self.padding = self.kernel_size // 2
        self.pad_token_id = int(model_cfg["pad_token_id"])

        self.byte_embedding = nn.Embedding(
            num_embeddings=int(model_cfg["vocab_size"]),
            embedding_dim=int(model_cfg["byte_embedding_dim"]),
            padding_idx=self.pad_token_id,
        )
        self.conv = nn.Conv1d(
            in_channels=int(model_cfg["byte_embedding_dim"]),
            out_channels=int(model_cfg["hidden_size"]),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.transformer = ModernBertModel(ModernBertConfig(**model_cfg))

    def _reduce_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        kernel = torch.ones(
            1,
            1,
            self.kernel_size,
            device=attention_mask.device,
            dtype=torch.float32,
        )
        reduced = F.conv1d(
            attention_mask.unsqueeze(1).float(),
            kernel,
            stride=self.stride,
            padding=self.padding,
        )
        return (reduced.squeeze(1) > 0).long()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> BaseModelOutput:
        byte_embeddings = self.byte_embedding(input_ids)
        byte_embeddings = byte_embeddings * attention_mask.unsqueeze(-1).float()

        conv_input = byte_embeddings.transpose(1, 2)
        grouped_embeddings = self.conv(conv_input).transpose(1, 2)
        reduced_mask = self._reduce_attention_mask(attention_mask)
        grouped_embeddings = grouped_embeddings * reduced_mask.unsqueeze(-1).float()

        hidden = self.transformer(
            inputs_embeds=grouped_embeddings,
            attention_mask=reduced_mask,
        ).last_hidden_state
        return BaseModelOutput(last_hidden_state=hidden)


class CnnByteModernBertEncoder(nn.Module):
    """JEPA encoder with a CNN front-end that groups local byte patterns."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = CnnByteBackbone(cfg=cfg)

    def forward(
        self,
        global_input_ids: torch.Tensor,
        global_attn_mask: torch.Tensor,
        local_input_ids: torch.Tensor,
        local_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_global_views, global_length = global_input_ids.shape
        _, num_local_views, local_length = local_input_ids.shape

        flat_global_ids = global_input_ids.view(batch_size * num_global_views, global_length)
        flat_global_mask = global_attn_mask.view(batch_size * num_global_views, global_length)
        flat_local_ids = local_input_ids.view(batch_size * num_local_views, local_length)
        flat_local_mask = local_attn_mask.view(batch_size * num_local_views, local_length)

        global_hidden = self.encoder(
            input_ids=flat_global_ids,
            attention_mask=flat_global_mask,
        ).last_hidden_state
        local_hidden = self.encoder(
            input_ids=flat_local_ids,
            attention_mask=flat_local_mask,
        ).last_hidden_state

        reduced_global_mask = self.encoder._reduce_attention_mask(flat_global_mask)
        reduced_local_mask = self.encoder._reduce_attention_mask(flat_local_mask)

        _, _, hidden_size = global_hidden.shape

        z_global = mean_pooling(global_hidden, reduced_global_mask)
        z_local = mean_pooling(local_hidden, reduced_local_mask)

        z_global = z_global.view(batch_size, num_global_views, hidden_size)
        z_local = z_local.view(batch_size, num_local_views, hidden_size)
        return torch.cat([z_global, z_local], dim=1)
