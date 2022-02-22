import torch
import clip
import torch.nn as nn

class clip_model(nn.Module):
    def __init__(self, image_dim, text_dim):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32")
        # self.compose_linear = nn.Linear(image_dim+text_dim, image_dim)

    def forward(self, image, text):
        image_feature = self.clip_model.encode_image(image) # batch x 512
        text_feature = self.clip_model.encode_text(clip.tokenize(text).cuda()) # batch x 512
        compose_feature = torch.cat((image_feature, text_feature), dim=-1)
        # compose_feature = self.compose_linear(compose_feature)
        return compose_feature 