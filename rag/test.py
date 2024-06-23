import gradio as gr
import random
import time

choices = ["one"]

def submit_button(input):
    global choices
    choices.append(input)
    return ' '.join(choices)

def respond(message, chat_history):
    global choices
    bot_message = random.choice(choices)
    chat_history.append((message, bot_message))
    time.sleep(2)
    return "", chat_history

with gr.Blocks() as demo:
    choice = gr.Textbox(label='Add one more choice')
    output2 = gr.Textbox(label="output")
    submit = gr.Button("Submit")
    submit.click(fn=submit_button, inputs=choice, outputs=output2)
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])



    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()