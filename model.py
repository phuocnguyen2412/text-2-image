import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    kết hợp thông tin từ text (caption) và noise (random vector) để sinh ảnh.
    noise là tín hiệu ngẫu nhiên giúp đa dạng hóa ảnh.
    """
    def __init__(self, noise_size, feature_size, num_channels, embedding_size, reduced_dim_size):
        super(Generator, self).__init__()
        self.noise_size = noise_size


        # Giảm chiều của embeded_caption từ 768 -> 256
        self.textEncoder = nn.Sequential(
            nn.Linear(embedding_size, reduced_dim_size),  #768 -> 256
            nn.BatchNorm1d(num_features=reduced_dim_size), #Chuẩn hóa đầu ra của Linear layer giúp ổn định quá trình training và đẩy nhanh hội tụ tránh vấn đề exploding/vanishing gradient
            nn.LeakyReLU(negative_slope=0.2, inplace=True) #Khi x âm, nó vẫn cho gradient khác 0 (slope là 0.2) -> Tránh chết neuron, gradient vẫn chạy ngược về học tốt hơn
        )

        self.upsamplingBlock = nn.Sequential(
            #256 + 100 -> 1024
            nn.ConvTranspose2d(noise_size + reduced_dim_size, feature_size * 8,4,1,0,bias=False), # phóng to ản và học cách lấp thông tin vào các pixel bị "mở rộng".
            nn.BatchNorm2d(num_features=feature_size * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #1024 -> 512
            nn.ConvTranspose2d(feature_size * 8, feature_size * 4,4,2,1,bias=False), # Layer này sẽ giảm số channels Đồng thời phóng to kích thước chiều cao và chiều rộng của feature map do stride = 2
            nn.BatchNorm2d(num_features=feature_size * 4),
            nn.ReLU(inplace=True),

            #512 -> 256
            nn.ConvTranspose2d(feature_size * 4, feature_size * 2,4,2,1,bias=False),
            nn.BatchNorm2d(num_features=feature_size * 2),
            nn.ReLU(inplace=True),

            #256 -> 128
            nn.ConvTranspose2d(feature_size * 2, feature_size,4,2,1,bias=False),
            nn.BatchNorm2d(num_features=feature_size),
            nn.ReLU(inplace=True),

            #128 -> 3
            nn.Conv2d(feature_size, num_channels,4,2,1, bias=False), # Giảm số kênh (channels) từ 128 → 3, tức là chuyển từ feature map sang ảnh RGB (3 channels) Đồng thời giảm kích thước không gian (chiều cao và chiều rộng) thêm một lần nữa.
            nn.Tanh() # Giới hạn giá trị pixel trong khoảng [-1, 1] Vì ảnh thật (real images) sau khi normalize thường được chuyển về [-1, 1]
        )

    def forward(self, noise, text_embedding):
        encoded_text = self.textEncoder(text_embedding) #[batch_size, reduced_dim_size]
        concat_input = torch.cat([noise, encoded_text], dim=1).unsqueeze(2).unsqueeze(2) #[batch_size, noise_size + reduced_dim_size, 1, 1]
        output = self.upsamplingBlock(concat_input) #[batch_size, num_channels, 16, 16]
        return output


class Discriminator(nn.Module):
    """
    Nhận ảnh đầu vào và embedding caption.
    Encode cả hai (ảnh và caption).
    Kết hợp thông tin giữa ảnh và caption, kiểm tra xem ảnh có khớp với caption hay không, và có phải ảnh thật hay fake từ Generator hay không.
    """
    def __init__(self, num_channels, feature_size, embedding_size, reduced_dim_size):
        super(Discriminator, self).__init__()
        self.reduced_dim_size = reduced_dim_size

        #Trích xuất đặc trưng chi tiết từ ảnh, giúp mạng phân biệt ảnh có "giống thật" và "khớp caption" không.
        self.imageEncoder = nn.Sequential(
            # (B, 3, H, W) → (B, 128, H/2, W/2)
            #3 -> 128
            nn.Conv2d(num_channels, feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            #128 -> 128
            nn.Conv2d(feature_size, feature_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            #128 -> 256
            nn.Conv2d(feature_size, feature_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            #256 -> 512
            nn.Conv2d(feature_size * 2, feature_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            #512 -> 1024 # (B, 1024, H/32, W/32)
            nn.Conv2d(feature_size * 4, feature_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Giảm chiều embedding từ 768 -> 256
        self.textEncoder = nn.Sequential(
            nn.Linear(embedding_size, reduced_dim_size),
            nn.BatchNorm1d(reduced_dim_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.finalBlock = nn.Sequential(
            nn.Conv2d(feature_size * 8 + reduced_dim_size, 1, 4, 1, 0, bias=False), #  (B, 1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding):
        """
        :param image: (B, 3, H, W)
        :param text_embedding: (B, 768)
        :return:
        """
        image_encoded = self.imageEncoder(image) # (B, 1024, H/32, W/32)
        text_encoded = self.textEncoder(text_embedding)#  (B, 256)

        #Ghép thông tin ảnh và text lại cùng không gian. Ý tưởng là: "Mỗi vùng ảnh" biết "thông tin text", để kiểm tra xem ảnh có match với caption không.
        replicated = text_encoded.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        combined = torch.cat([image_encoded, replicated], dim=1) # (B, 1024 + 256, 4, 4)

        output = self.finalBlock(combined).view(-1,1) # Chuyển về tensor shape (B, 1)
        return output.view(-1, 1), image_encoded