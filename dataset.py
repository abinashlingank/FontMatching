from PIL import Image, ImageDraw, ImageFont
import os

font_styles = {
    'Arimo': ['Arimo-Italic-VariableFont_wght', 'Arimo-VariableFont_wght'],
    'Dancing_Script': ['DancingScript-VariableFont_wght'],
    'Fredoka': ['Fredoka-VariableFont_wdth,wght'],
    'Noto_Sans': ['NotoSans-Italic-VariableFont_wdth,wght', 'NotoSans-VariableFont_wdth,wght'],
    'Open_Sans': ['OpenSans-Italic-VariableFont_wdth,wght', 'OpenSans-VariableFont_wdth,wght'],
    'Oswald': ['Oswald-VariableFont_wght'],
    'Patua_One': ['PatuaOne-Regular'],
    'PT_Serif': ['PTSerif-Italic', 'PTSerif-Regular', 'PTSerif-Bold', 'PTSerif-BoldItalic'],
    'Roboto': ['Roboto-BlackItalic', 'Roboto-Regular', 'Roboto-ThinItalic', 'Roboto-Black'],
    'Ubuntu': ['Ubuntu-Light', 'Ubuntu-Regular', 'Ubuntu-Bold', 'Ubuntu-Medium']
}

fonts_data = []

# Directory to save generated images
for folder, fonts in font_styles.items():
    output_dir = f"/content/dataset/{folder}"
    os.makedirs(output_dir, exist_ok=True)
    fonts_data.append(folder)

    # Text to overlay
    text = ["Hello,World!", "World!", "Hello", "World Hello!", ",World", "World!!!!, Hello", "Hello, !!World",
            "!!!Hello,", "Hello@ World", "World@ ", "Hello World", "HELLO WORLD"]

    # Image and font size
    image_size = (400, 100)
    font_size = 36

    # Generate synthetic images
    for word in text:
        for font_name in fonts:
            # Create blank image
            img = Image.new("RGB", image_size, color="white")

            # Load font
            font_path = f"/content/drive/MyDrive/Fonts/{folder}/{font_name}.ttf"
            font = ImageFont.truetype(font_path, font_size)

            # Draw text on image
            draw = ImageDraw.Draw(img)
            draw.text((50, 30), word, fill="black", font=font)

            # Save image
            img.save(os.path.join(output_dir, f"{font_name}_{word}.png"))

print("Synthetic data generation complete.")
print(fonts_data)