import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
#use this command to run the script:
#python flux_test.py

# 优化：删除不再需要的变量和清理缓存
torch.cuda.empty_cache()  # 清理 CUDA 缓存

# 方案1: 使用完整的4bit量化模型（推荐）
repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

# 加载完整pipeline，包含text encoder
pipe = Flux2Pipeline.from_pretrained(
    repo_id, 
    torch_dtype=torch_dtype
).to(device)


# 使用 torch.no_grad() 以避免计算梯度并减少内存使用
with torch.no_grad():

    prompt = """Full body photo, realistic candid style, wide shot, high resolution, high detail, natural look, soft daylight with cool undertones, no golden hour. Subject: a young, slender Asian woman, cute and youthful with long straight black hair, smooth texture, falling naturally over her shoulders and down her back. Her face has soft oval features, with gentle jawline, subtle high cheekbones, and a balanced appearance. Her eyes are large but proportionate to her face, almond-shaped with natural catchlight, and the gaze is calm and neutral, not exaggerated. The eyes have a subtle, thoughtful look, giving her a sense of quiet confidence and introspection. Her nose is small and well-defined, with a smooth bridge and soft, rounded tip, adding balance to her features. Her full natural lips are slightly closed, with a gentle curve at the corners, giving her a serene and calm expression. The lips have a natural soft pink hue, not overly defined. 

Her skin is fair, smooth, and realistic, with a healthy glow, showing subtle natural pores and light skin texture on the face and neck, making it appear genuine and lifelike. No excessive smoothing, realistic texture on the skin, slight blush on the cheeks, and soft sheen across her cheekbones and nose. No exaggerated features, no plastic-like skin.

Outfit: simple white short-sleeve blouse, slightly translucent, lightweight cotton fabric, minimalistic design, subtly see-through in natural light. High-waisted navy blue skirt, smooth fabric, just above the knee, accentuating her slender legs. White crew socks slightly loose at the ankles, visible rib texture. Casual black sneakers, matte finish, simple laces, comfortable and stylish.

Pose: sitting sideways on yellow stadium seats, one knee bent naturally and the other leg extended forward and downward. Her right foot rests on the **connection point between the backrest and seat cushion** of the yellow plastic seat in front of her. The foot is lightly placed, not flat on the seat back, ensuring a relaxed, natural pose. Legs are long and slender, smooth calves and thighs, ankles delicate, emphasizing her natural figure. Hands rest gently on her lap, one lightly touching her skirt.

Prop: small bouquet of flowers placed on the empty yellow seat next to her (not in her hands), adding a touch of nature.

Background: modern outdoor stadium with vibrant red and yellow seats arranged in rows. Midground features an orange painted concrete wall with blue-gray railings. Upper background shows a large steel roof structure with curved beams and glass facade. Empty stadium, no crowd, clean environment. Strong perspective lines from the seats leading toward her, creating depth and space.

Composition: wide angle, low camera angle from seat level, full body shot, subject slightly to the right. Colorful seats frame her in the shot. Background is slightly blurred but detailed, no heavy bokeh, sharp geometry with minimal distortion. The lighting is soft, casting a natural cool light over her figure, realistic contrast, no HDR, film-like color grading.

Natural, serene, youthful vibe with a fresh, dynamic atmosphere, capturing a candid moment with natural sunlight and clear details."""
    
    negative_prompt = """close-up, half body, cropped legs, missing feet, bokeh heavy, studio lighting, glamour, fashion editorial,
anime, illustration, CGI, plastic skin, doll face, over-smooth skin, thick legs, muscular legs, exaggerated curves,
extra limbs, extra fingers, deformed hands, bad anatomy, high heels, boots, barefoot,
holding bouquet, bouquet in hand, sunset golden hour, warm orange lighting"""

    image = pipe(
        prompt=prompt,
        generator=torch.Generator(device=device).manual_seed(42),
        num_inference_steps=50,
        guidance_scale=4.5,
        width=1920,  # 适当减少分辨率来节省内存
        height=1080,  # 适当减少分辨率来节省内存
    ).images[0]

# 保存生成的图像
image.save("flux2_output2.png")
print("图像已保存到 flux2_output2.png")