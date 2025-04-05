import os
from openai import OpenAI
import requests
from datetime import datetime
api_key = ""


class ImageGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=api_key)
        self.output_dir = 'generated_images'
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_images(self, prompt, n=1, size="1024x1024", model="dall-e-3"):
        """
        Generate images using OpenAI's DALL-E
        """
        try:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size
            )
            return response.data
        except Exception as e:
            print(f"Error generating images: {str(e)}")
            return None

    def download_image(self, image_url, prompt):
        """
        Download generated image
        """
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.output_dir}/{timestamp}_{prompt[:30]}.png"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename
            return None
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            return None

    def save_image(self, prompt, num_images):
        # Generate images
        images = self.generate_images(prompt, n=num_images)
        if images:
            for idx, image in enumerate(images, 1):
                filename = self.download_image(image.url, prompt)
                if filename:
                    print(f"Image {idx} saved to {filename}")
                else:
                    print(f"Failed to download image {idx}")

def main():
    # Replace with your OpenAI API key
    generator = ImageGenerator()

    # Example usage
    prompt = input("Enter your image prompt: ")
    num_images = int(input("How many images (1-10)? "))

    generator.save_image(prompt, num_images)


if __name__ == "__main__":
    main()