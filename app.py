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
  Set the main character as a chinese child if it is not specified.
  Remove anything that is inappropriate for children.
  Do not show any revealing clothing.
  Do not dipict any violence or weapon or horror.
  Do not dispict any smoking, alcohol, or drug.
  Only reply with the new prompt.

  If the original input contains the following words, use the suggested translation: \n
    1. 鸡蛋卷: crispy rolled wafer pastry
    2. 肉干: BBQ dried meat jerky slice in shaqe shape and dark red in color
    3. 欢欢: young chinese toddler girl
    4. 小雨: young chinese toddler girl
    5. 小云: young chinese toddler girl
    6. 康康: young chinese toddler boy
    7. 巴刹: wet market
    8. 组屋: 10 storey, multicolored residental flats with Singapore national flags hanging outside some windows
    9. 字宝宝: card game to learn Chinese words
    10 冲凉房: shower cubicle

  Here is the original input: \n
"""


def stablediffusion(prompt):
  url = "https://stablediffusionapi.com/api/v3/text2img"

  payload = json.dumps({
    "key": "zUHkLl4AxUM19kd2zW4gjKZPoSug3qnB18iJ4toFQBwi4BICBTYnhwYkBU0O",
    "prompt": prompt, "negative_prompt": None, "width": "512", "height": "512",
    "samples": "1", "num_inference_steps": "20", "seed": None, "guidance_scale": 7.5,
    "safety_checker": "yes", "multi_lingual": "no", "panorama": "no", "self_attention": "no",
    "upscale": "no", "embeddings_model": None, "webhook": None, "track_id": None
  })

  response = requests.request("POST", url, headers= {'Content-Type': 'application/json'}, data=payload).json()

  print(response)
  return response['output'][0]

def generateImg(text):

  markdown = 'FOR DEV & DEBUG ONLY (This segment will not be visible to frontend users) \n\n'

  markdown = markdown + 'Layer 1: Original Input: \n\n' + text + '\n\n'

  result = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[ {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt + text}]
  )

  ai_prompt = result.choices[0].message.content 

  markdown = markdown + 'Layer 2: AI LLM Translation & Content Filtering Algorithm: \n\n' + ai_prompt + '\n\n'

  # img_url = stablediffusion(ai_prompt)

  img = client.images.generate(
    model="dall-e-3",
    prompt=ai_prompt,
    size="1024x1024",
    quality="standard",
    n=1,
  )
  print(img)

  enhanded_prompt = str(img.data[0].revised_prompt)

  markdown = markdown + 'Layer 3: Vision Model Enhanced Prompt: \n\n ' + enhanded_prompt + '\n\n Layer 4: DALLE Generated Image: \n\n'

  img_url = img.data[0].url

  print(ai_prompt)
  print(img_url)

  return [markdown, img_url]


from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")



def visualQA(img, prompt):

  markdown = '图片要求: ' + prompt + '\n\n'
  
  raw_image = Image.open(img).convert('RGB')
  inputs = processor(raw_image, return_tensors="pt")
  out = model.generate(**inputs, min_length=10, max_length=200)
  description = processor.decode(out[0], skip_special_tokens=True)
  print('description', description) 
  markdown = markdown + 'Layer 1: AI Image Filtering & Captioning: \n\n ' + description + '\n\n'

  visualQA_prompt = "First translate this description of an image to Chinese: " + description + ". Then, based on this description of an image, determine whether it fits this requirement: " + prompt + 'Reply in this formate: 1) 图片内容: ... \n\n 2) 结果: 图片符合要求 / 图片不符合要求' 

  result = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[ {"role": "system", "content": 'You are a primary school teacher'},
              {"role": "user", "content": visualQA_prompt }]
  ).choices[0].message.content 

  markdown = markdown + 'Layer 2: AI 图片判断结果: \n\n' + result + '\n\n'

  return markdown

with gr.Blocks(title='ImageGen-3ESystems') as demo:

  gr.Markdown("""# ImageGen-3ESystems
              A proof of concept demo created by [3E Systems](https://3esystems.sg). All rights reserved.""")

  with gr.Tab("造句生图"):
    inp = gr.Textbox(label='学生造句', placeholder="一只小狗在路边玩耍...")
    btn = gr.Button(value="AI 生成")
    out = gr.Markdown()

    img = gr.Image(type='filepath', label="AI 生成图", show_download_button=True, interactive=False)

    btn.click(generateImg, inputs=[inp], outputs=[out, img])

  with gr.Tab("图像识别"):
    inp_prompt = gr.Textbox(label='图片要求 （老师输入）', placeholder="橙子/衣柜/散步/玩耍/...")
    inp_img = gr.Image(type='filepath', label="图片（学生上传/拍照）")
    out = gr.Markdown()
    btn2 = gr.Button(value="AI 识别")

    btn2.click(visualQA, inputs=[inp_img, inp_prompt], outputs=[out])

demo.launch()