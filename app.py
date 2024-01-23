import gradio as gr
from openai import OpenAI
import requests
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

system_prompt = """
  You are a prompt engineer for a text to image generation AI model.
  Your job is to ensure that the model generates appropriate images based on the prompt.

"""

user_prompt = """
  Translate the original input from Chinese to English.
  Enhance the original input to make it more suitable for the model to generate a cute, high-quality image.
  If the input is too vague, add more details.
  Remove anything that is inappropriate for children.
  Only reply with the new prompt.
  Here is the original input: \n
"""


def stablediffusion(prompt):

  url = "https://stablediffusionapi.com/api/v3/text2img"

  payload = json.dumps({
    "key": "zUHkLl4AxUM19kd2zW4gjKZPoSug3qnB18iJ4toFQBwi4BICBTYnhwYkBU0O",
    "prompt": prompt,
    "negative_prompt": None,
    "width": "512",
    "height": "512",
    "samples": "1",
    "num_inference_steps": "20",
    "seed": None,
    "guidance_scale": 7.5,
    "safety_checker": "yes",
    "multi_lingual": "no",
    "panorama": "no",
    "self_attention": "no",
    "upscale": "no",
    "embeddings_model": None,
    "webhook": None,
    "track_id": None
  })

  response = requests.request("POST", url, headers= {'Content-Type': 'application/json'}, data=payload).json()

  print(response)

  # print(response.json)
  print(response['output'][0])
  print(response['output'][0].replace('\/', '/'))
  return response['output'][0]


def dalle(prompt):
  img = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
  )
  print(img)
  img_url = img.data[0].url

  return img_url

def generate(text):
  result = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt + text}
    ]
  )

  ai_prompt = result.choices[0].message.content 



  # img_url = stablediffusion(ai_prompt)
  img_url = dalle(ai_prompt)

  print(ai_prompt)
  print(img_url)

  return [ai_prompt, img_url]


with gr.Blocks() as demo:

  inp = gr.Textbox(placeholder="一只小狗在路边玩耍...")
  btn = gr.Button(value="Generate")

  out = gr.Markdown()
  img = gr.Image(type='filepath', show_download_button=True, interactive=False)

  btn.click(generate, inputs=[inp], outputs=[out, img])


demo.launch()