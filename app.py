import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

client = OpenAI()

system_prompt = os.getenv("SYSTEM_PROMPT")
user_prompt = os.getenv("USER_PROMPT")
retry_prompt = os.getenv("RETRY_PROMPT")
vocab_dict = os.getenv("VOCAB_DICT").split(',')

def generateImg(text, retry=False):

  markdown = 'FOR DEV & DEBUG ONLY (This segment will not be visible to frontend users) \n\n'
  markdown = markdown + 'Layer 1: Original Input: \n\n' + text + '\n\n'

  for vocab in vocab_dict:
    text = text.replace(vocab[: vocab.index(':')].strip(), ' ' + vocab + ' ')

  print(text)

  if retry:
    print('retrying')
    llm_prompt = retry_prompt + text
  else:
    llm_prompt = user_prompt + text

  result = client.chat.completions.create(model=os.getenv("LLM_MODEL"), messages=[ {"role": "system", "content": system_prompt},{"role": "user", "content": llm_prompt}])
  ai_prompt = result.choices[0].message.content 
  markdown = markdown + 'Layer 2: AI LLM Translation & Content Filtering Algorithm: \n\n' + ai_prompt + '\n\n'
  img = client.images.generate(model=os.getenv('IMG_MODEL'), prompt= 'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS: '+ai_prompt, size="1024x1024", quality="standard", n=1)
  markdown = markdown + '\n\n Layer 4: DALLE Generated Image: \n\n'
  img_url = img.data[0].url
  print(ai_prompt)
  print(img_url)
  return [markdown, img_url]


with gr.Blocks(title='ImageGen-3ESystems') as demo:


  gr.Markdown("""# ImageGen-3ESystems
              A proof of concept demo created by [3E Systems](https://3esystems.sg). All rights reserved.""")

  with gr.Tab("造句生图"):
    inp = gr.Textbox(label='学生造句', placeholder="一只小狗在路边玩耍...")
    btn = gr.Button(value="AI 生成")
    out = gr.Markdown()
    retry = gr.State(True)

    img = gr.Image(type='filepath', label="AI 生成图", show_download_button=True, interactive=False)

    btn.click(generateImg, inputs=[inp], outputs=[out, img])

    retryBtn = gr.Button(value="再生成一次")
    retryBtn.click(generateImg, inputs=[inp, retry], outputs=[out, img])

demo.launch()