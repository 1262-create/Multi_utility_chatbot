#!/usr/bin/env python3
"""
Cloud-Based Image Generation Alternatives
No local ComfyUI setup required!
"""

import os
import base64
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# OPTION 1: OPENAI DALL-E (Most Popular & Easy)
# =============================================================================

class OpenAIImageGenerator:
    """Generate images using OpenAI's DALL-E API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY in .env")

    def generate_image(self, prompt: str, size: str = "1024x1024", style: str = "natural") -> Dict[str, Any]:
        """
        Generate image using DALL-E

        Args:
            prompt: Text description
            size: "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
            style: "natural" or "vivid"

        Returns:
            Dictionary with image data
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "dall-e-3",  # or "dall-e-2"
                "prompt": prompt,
                "n": 1,
                "size": size,
                "style": style
            }

            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                image_url = result["data"][0]["url"]

                # Download the image
                image_response = requests.get(image_url, timeout=30)
                if image_response.status_code == 200:
                    image_data = image_response.content
                    image_base64 = base64.b64encode(image_data).decode('utf-8')

                    return {
                        "status": "success",
                        "message": "Image generated successfully with DALL-E!",
                        "image_base64": image_base64,
                        "image_data": image_data,
                        "provider": "OpenAI DALL-E",
                        "model": "dall-e-3",
                        "prompt": prompt,
                        "size": size,
                        "style": style
                    }
                else:
                    return {"status": "error", "message": "Failed to download generated image"}
            else:
                error = response.json()
                return {
                    "status": "error",
                    "message": f"DALL-E API error: {error.get('error', {}).get('message', 'Unknown error')}"
                }

        except Exception as e:
            return {"status": "error", "message": f"OpenAI generation failed: {str(e)}"}


# =============================================================================
# OPTION 2: STABILITY AI (Cost Effective)
# =============================================================================

class StabilityImageGenerator:
    """Generate images using Stability AI API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("STABILITY_API_KEY")
        if not self.api_key:
            raise ValueError("Stability AI API key required. Set STABILITY_API_KEY in .env")

    def generate_image(self, prompt: str, style: str = "enhance", aspect_ratio: str = "1:1") -> Dict[str, Any]:
        """
        Generate image using Stability AI

        Args:
            prompt: Text description
            style: "enhance", "anime", "photographic", "digital-art", "comic-book", "fantasy-art", "line-art", "analog-film", "neon-punk", "isometric", "low-poly", "origami", "modeling-compound", "cinematic", "3d-model", "pixel-art", "tile-texture"
            aspect_ratio: "16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"

        Returns:
            Dictionary with image data
        """
        try:
            url = f"https://api.stability.ai/v2beta/stable-image/generate/core"

            headers = {
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*"
            }

            # Map aspect ratios
            aspect_map = {
                "1:1": "1:1",
                "16:9": "16:9",
                "9:16": "9:16",
                "21:9": "21:9",
                "2:3": "2:3",
                "3:2": "3:2",
                "4:5": "4:5",
                "5:4": "5:4",
                "9:21": "9:21"
            }

            data = {
                "prompt": prompt,
                "aspect_ratio": aspect_map.get(aspect_ratio, "1:1"),
                "style_preset": style,
                "output_format": "png"
            }

            response = requests.post(url, headers=headers, files={"none": ""}, data=data, timeout=60)

            if response.status_code == 200:
                image_data = response.content
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                return {
                    "status": "success",
                    "message": "Image generated successfully with Stability AI!",
                    "image_base64": image_base64,
                    "image_data": image_data,
                    "provider": "Stability AI",
                    "model": "Stable Image Core",
                    "prompt": prompt,
                    "style": style,
                    "aspect_ratio": aspect_ratio
                }
            else:
                try:
                    error = response.json()
                    return {
                        "status": "error",
                        "message": f"Stability AI error: {error.get('message', 'Unknown error')}"
                    }
                except:
                    return {"status": "error", "message": f"Stability AI API error: {response.status_code}"}

        except Exception as e:
            return {"status": "error", "message": f"Stability AI generation failed: {str(e)}"}


# =============================================================================
# OPTION 3: REPLICATE (Cloud ComfyUI - No Local Setup!)
# =============================================================================

class ReplicateImageGenerator:
    """Generate images using Replicate's cloud ComfyUI models"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_key:
            raise ValueError("Replicate API token required. Set REPLICATE_API_TOKEN in .env")

        # Popular ComfyUI models on Replicate
        self.models = {
            "marketing_poster": "zsxkib/comfyui:1.0.0",  # Generic ComfyUI
            "social_media": "zsxkib/comfyui:1.0.0",
            # You can add specific fine-tuned models here
        }

    def generate_image(self, prompt: str, style: str = "marketing_poster", aspect_ratio: str = "1:1") -> Dict[str, Any]:
        """
        Generate image using Replicate's ComfyUI models

        Args:
            prompt: Text description
            style: "marketing_poster" or "social_media"
            aspect_ratio: "1:1" or "9:16"

        Returns:
            Dictionary with image data
        """
        try:
            import replicate

            # Set API token
            os.environ["REPLICATE_API_TOKEN"] = self.api_key

            model = self.models.get(style, self.models["marketing_poster"])

            # Set dimensions based on aspect ratio
            if aspect_ratio == "1:1":
                width, height = 1024, 1024
            elif aspect_ratio == "9:16":
                width, height = 768, 1344  # Portrait
            else:
                width, height = 1024, 1024

            # Run the model
            output = replicate.run(
                model,
                input={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "scheduler": "DPMSolverMultistep"
                }
            )

            if output and len(output) > 0:
                # Output is usually a URL or list of URLs
                image_url = output[0] if isinstance(output, list) else output

                # Download the image
                image_response = requests.get(image_url, timeout=30)
                if image_response.status_code == 200:
                    image_data = image_response.content
                    image_base64 = base64.b64encode(image_data).decode('utf-8')

                    return {
                        "status": "success",
                        "message": "Image generated successfully with Replicate ComfyUI!",
                        "image_base64": image_base64,
                        "image_data": image_data,
                        "provider": "Replicate",
                        "model": "ComfyUI Cloud",
                        "prompt": prompt,
                        "style": style,
                        "aspect_ratio": aspect_ratio
                    }
                else:
                    return {"status": "error", "message": "Failed to download generated image"}
            else:
                return {"status": "error", "message": "No output received from Replicate"}

        except Exception as e:
            return {"status": "error", "message": f"Replicate generation failed: {str(e)}"}


# =============================================================================
# OPTION 4: HUGGING FACE FREE INFERENCE (Completely Free!)
# =============================================================================

class HuggingFaceImageGenerator:
    """Generate images using free Hugging Face inference API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        # Updated to use the free Hugging Face Inference API endpoint
        self.api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

    def generate_image(self, prompt: str, style: str = "base", aspect_ratio: str = "1:1") -> Dict[str, Any]:
        """
        Generate image using Hugging Face free inference

        Args:
            prompt: Text description
            style: Not used (limited customization)
            aspect_ratio: Not used (fixed 512x512)

        Returns:
            Dictionary with image data
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

            # Enhance prompt for better results
            enhanced_prompt = f"{prompt}, high quality, professional, detailed"

            data = {
                "inputs": enhanced_prompt,
                "parameters": {
                    "width": 512,
                    "height": 512,
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5
                }
            }

            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)

            if response.status_code == 200:
                image_data = response.content
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                return {
                    "status": "success",
                    "message": "Image generated successfully with Hugging Face!",
                    "image_base64": image_base64,
                    "image_data": image_data,
                    "provider": "Hugging Face",
                    "model": "Stable Diffusion 2.1",
                    "prompt": prompt,
                    "note": "Free tier has queue times and rate limits"
                }
            else:
                try:
                    error = response.json()
                    return {
                        "status": "error",
                        "message": f"Hugging Face error: {error.get('error', 'Unknown error')}"
                    }
                except:
                    return {"status": "error", "message": f"Hugging Face API error: {response.status_code}"}

        except Exception as e:
            return {"status": "error", "message": f"Hugging Face generation failed: {str(e)}"}


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

class CloudImageGenerator:
    """
    Unified interface for different cloud image generation services.
    Automatically chooses the best available service.
    """

    def __init__(self):
        self.generators = {}

        # Initialize available generators
        try:
            if os.getenv("OPENAI_API_KEY"):
                self.generators["openai"] = OpenAIImageGenerator()
        except:
            pass

        try:
            if os.getenv("STABILITY_API_KEY"):
                self.generators["stability"] = StabilityImageGenerator()
        except:
            pass

        try:
            if os.getenv("REPLICATE_API_TOKEN"):
                self.generators["replicate"] = ReplicateImageGenerator()
        except:
            pass

        # Hugging Face is always available (with or without API key)
        self.generators["huggingface"] = HuggingFaceImageGenerator()

        # Priority order: OpenAI (best quality) -> Stability (cost effective) -> Replicate -> Hugging Face (free)
        self.priority = ["openai", "stability", "replicate", "huggingface"]

    def generate_image(self, prompt: str, style: str = "marketing_poster", aspect_ratio: str = "1:1", provider: str = "auto") -> Dict[str, Any]:
        """
        Generate image using the best available cloud service

        Args:
            prompt: Text description
            style: "marketing_poster" or "social_media"
            aspect_ratio: "1:1" or "9:16"
            provider: "auto", "openai", "stability", "replicate", "huggingface"

        Returns:
            Dictionary with image data
        """
        if not self.generators:
            return {
                "status": "error",
                "message": "No image generation services configured. Please set API keys in .env file."
            }

        # Choose provider
        if provider == "auto":
            for p in self.priority:
                if p in self.generators:
                    provider = p
                    break
        elif provider not in self.generators:
            return {
                "status": "error",
                "message": f"Provider '{provider}' not available. Available: {list(self.generators.keys())}"
            }

        # Get the generator
        generator = self.generators[provider]

        # Map style to provider-specific parameters
        if provider == "openai":
            size_map = {"1:1": "1024x1024", "9:16": "1024x1792"}
            size = size_map.get(aspect_ratio, "1024x1024")
            style_param = "vivid" if style == "social_media" else "natural"
            return generator.generate_image(prompt, size=size, style=style_param)

        elif provider == "stability":
            style_map = {
                "marketing_poster": "enhance",
                "social_media": "photographic"
            }
            style_param = style_map.get(style, "enhance")
            return generator.generate_image(prompt, style=style_param, aspect_ratio=aspect_ratio)

        elif provider == "replicate":
            return generator.generate_image(prompt, style=style, aspect_ratio=aspect_ratio)

        elif provider == "huggingface":
            return generator.generate_image(prompt, style=style, aspect_ratio=aspect_ratio)

        else:
            return {"status": "error", "message": f"Unknown provider: {provider}"}


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

# Global instance
cloud_generator = CloudImageGenerator()

def generate_marketing_image_cloud(
    prompt: str,
    style: str = "marketing_poster",
    aspect_ratio: str = "1:1",
    provider: str = "auto"
) -> Dict[str, Any]:
    """
    Generate marketing images using cloud services (no local ComfyUI required!)

    Args:
        prompt: Description of the image you want to generate
        style: Style of image - 'marketing_poster' or 'social_media'
        aspect_ratio: Aspect ratio - '1:1' (square) or '9:16' (portrait)
        provider: Cloud provider - 'auto', 'openai', 'stability', 'replicate', 'huggingface'

    Returns:
        Dictionary with generation results
    """
    return cloud_generator.generate_image(prompt, style, aspect_ratio, provider)


# =============================================================================
# SETUP INSTRUCTIONS
# =============================================================================

SETUP_INSTRUCTIONS = """
🚀 Cloud Image Generation Setup (No Local ComfyUI Required!)

Choose your preferred provider and set the corresponding API key in your .env file:

1. 🎨 OpenAI DALL-E (Best Quality, $0.08-0.12 per image)
   - Set: OPENAI_API_KEY=your_openai_key
   - Pros: Best quality, fast, reliable
   - Cons: Paid service

2. 🖼️ Stability AI (Cost Effective, $0.02-0.04 per image)
   - Set: STABILITY_API_KEY=your_stability_key
   - Pros: Affordable, good quality, style presets
   - Cons: Requires API key

3. ☁️ Replicate (Cloud ComfyUI, $0.005-0.01 per second)
   - Set: REPLICATE_API_TOKEN=your_replicate_token
   - Pros: Full ComfyUI power, customizable
   - Cons: Complex setup

4. 🤗 Hugging Face (Free, Rate Limited)
   - Set: HUGGINGFACE_API_KEY=your_huggingface_key (optional)
   - Pros: Completely free, no API key required
   - Cons: Queue times, lower quality, rate limits

📝 Example .env file:
OPENAI_API_KEY=sk-your-openai-key-here
STABILITY_API_KEY=sk-your-stability-key-here
REPLICATE_API_TOKEN=r-your-replicate-token-here
HUGGINGFACE_API_KEY=hf-your-huggingface-key-here

🎯 Usage in chatbot:
"generate a marketing poster for my coffee shop"
"create a professional logo for my tech startup"
"design social media content for my restaurant"
"""