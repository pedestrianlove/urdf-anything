"""Multi-output DiT model: latent + param1/2/3 + motion_type heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit_triposg import TripoSGDiTModel


class MultiOutputDiTModel(TripoSGDiTModel):
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1024,
        use_cross_attention_2: bool = True,
        cross_attention_2_dim: int = 64,
        additional_output_dims: tuple = (3, 3, 2),
        shared_hidden_dim: int = 512,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            width=width,
            in_channels=in_channels,
            num_layers=num_layers,
            cross_attention_dim=cross_attention_dim,
            use_cross_attention_2=use_cross_attention_2,
            cross_attention_2_dim=cross_attention_2_dim,
        )
        self.additional_output_dims = additional_output_dims
        self.shared_hidden_dim = shared_hidden_dim
        self.proj_out = nn.Linear(self.inner_dim, self.shared_hidden_dim, bias=True)
        self._create_output_heads()

    def _create_output_heads(self):
        self.latent_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, self.out_channels),
        )
        self.param1_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.additional_output_dims[0]),
        )
        self.param1_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.param2_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.additional_output_dims[1]),
        )
        self.param2_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.param3_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.additional_output_dims[2]),
        )
        self.param3_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.motion_type_head = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.motion_type_attention = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        encoder_hidden_states_2=None,
        image_rotary_emb=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            attention_kwargs.pop("scale", 1.0)
        _, N, _ = hidden_states.shape
        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat([temb, hidden_states], dim=1)
        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs = (
                    {"use_reentrant": False} if torch.__version__ >= "1.11.0" else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_2,
                    temb,
                    image_rotary_emb,
                    skip,
                    attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=attention_kwargs,
                )
            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -N:]
        shared_features = self.proj_out(hidden_states)
        latent_output = self.latent_head(shared_features)
        param1_output = self.param1_head(shared_features)
        param2_output = self.param2_head(shared_features)
        param3_output = self.param3_head(shared_features)
        param1_attention_weights = F.softmax(self.param1_attention(shared_features), dim=1)
        param1_global = (param1_output * param1_attention_weights).sum(dim=1)
        param2_attention_weights = F.softmax(self.param2_attention(shared_features), dim=1)
        param2_global = (param2_output * param2_attention_weights).sum(dim=1)
        param3_attention_weights = F.softmax(self.param3_attention(shared_features), dim=1)
        param3_global = (param3_output * param3_attention_weights).sum(dim=1)
        motion_type_attention_weights = F.softmax(
            self.motion_type_attention(shared_features), dim=1
        )
        motion_type_global = (
            self.motion_type_head(shared_features) * motion_type_attention_weights
        ).sum(dim=1)
        if not return_dict:
            return (
                latent_output,
                param1_global,
                param2_global,
                param3_global,
                motion_type_global,
            )
        return {
            "latent": latent_output,
            "param1": param1_global,
            "param2": param2_global,
            "param3": param3_global,
            "motion_type": motion_type_global,
        }
