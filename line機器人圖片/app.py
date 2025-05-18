import sys
import configparser
import os
import azure.cognitiveservices.speech as speechsdk
import librosa
import tempfile
from PIL import Image

# Gemini API SDK
import google.generativeai as genai

from flask import Flask, request, abort, jsonify
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    AudioMessage
    
)
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient

# Config Parser
config = configparser.ConfigParser()
config.read('config.ini')

# Gemini API Settings
genai.configure(api_key=config["Gemini"]["API_KEY"])


text_translator = TextTranslationClient(
    credential = AzureKeyCredential(config["AzureTranslator"]["Key"]),
    endpoint = config["AzureTranslator"]["EndPoint"],
    region = config["AzureTranslator"]["Region"],
)

# 全域變數：儲存歷史紀錄
history_log = []

llm_role_description="""用繁體中文回答"""
# llm_role_description = """
# 你是餐廳顧問！
# 使用繁體中文來回答問題。
# 如果有人詢問餐點時，產生內容包含(餐點名稱->文化故事->所需材料->製作方法->過敏原警示->營養分析)
# """

# Use the model
from google.generativeai.types import HarmCategory, HarmBlockThreshold
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    },
    system_instruction=llm_role_description,
)

UPLOAD_FOLDER = "static"
app = Flask(__name__)

channel_access_token = config['Line']['CHANNEL_ACCESS_TOKEN']
channel_secret = config['Line']['CHANNEL_SECRET']
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

handler = WebhookHandler(channel_secret)

configuration = Configuration(
    access_token=channel_access_token
)

# Track uploaded images
is_image_uploaded = False
uploaded_images = []  # List to store image paths



@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(history_log), 200

@app.route("/history", methods=["DELETE"])
def delete_history():
    history_log.clear()
    return jsonify({"msg": "歷史紀錄已清除"}), 200

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=ImageMessageContent)
def message_image(event):
    global is_image_uploaded, uploaded_images

    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(
            message_id=event.message.id
        )
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_FOLDER, prefix="", delete=False
        ) as tf:
            tf.write(message_content)
            tempfile_path = tf.name

    # Append image path to the list
    uploaded_images.append(tempfile_path)
    is_image_uploaded = True

    # Respond with the current upload status
    finish_message = f"已收到第 {len(uploaded_images)} 張圖片，請繼續上傳或輸入問題開始處理。"
    history_log.append({
        "type": "image",
        "user": f"收到圖片：{event.message.id}",
        "bot": finish_message
    })
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=finish_message)],
            )
        )



@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    global is_image_uploaded, uploaded_images
    print(history_log)
    if is_image_uploaded:
        gemini_result = gemini_llm_sdk(event.message.text)
        #outputaudio_duration = azure_speech(gemini_result)or 0
        # Clear image upload state
        is_image_uploaded = False
        uploaded_images = []


        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=gemini_result)
                              #AudioMessage(originalContentUrl=config["Deploy"]["URL"]+"/static/outputaudio.wav", duration=outputaudio_duration)
                              ],
                )
            )
    else:
        # Handle regular text input
        gemini_result = gemini_llm_sdk(event.message.text)
        #outputaudio_duration = azure_speech(gemini_result)or 0
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=gemini_result)
                              #AudioMessage(originalContentUrl=config["Deploy"]["URL"]+"/static/outputaudio.wav", duration=outputaudio_duration)
                              ],
                )
            )
    history_log.append({
        "type": "text",
        "user": event.message.text,
        "bot": gemini_result
    })


def gemini_llm_sdk(user_input):
    try:
        if is_image_uploaded and uploaded_images:
            # Process multiple images
            uploaded_images_data = [Image.open(img_path) for img_path in uploaded_images]
            response = model.generate_content([user_input] + uploaded_images_data)
        else:
            if user_input == "清除歷史紀錄":
                delete_history()
                return "歷史紀錄已清除"
                
            content = user_input
            if len(history_log) == 0:
                response = model.generate_content(user_input)   
            else:
                content += history_log[-1]["user"] + "\n\n\n"+history_log[-1]["bot"]
                response = model.generate_content(content)
            

        print(f"Question: {user_input}")
        print(f"Answer: {response.text}")

        return response.text if hasattr(response, 'text') else str(response)

        target_language = "en"

            
        response_text = response.text if hasattr(response, 'text') else str(response)

        # 包裝成翻譯請求的 body
        body = [{"Text": response_text}] 
            
        response2 = text_translator.translate(
            body=body,  
            to_language=[target_language]  
        )
        print(response2)
        translation = response2[0].translations[0].text if response2 else None

        if translation:
            #azure_speech(response.text)
            #azure_speech(translation)
            return response.text+"\n\n\n翻譯結果:\n\n\n\n"+translation

    except Exception as e:
        print(e)
        return "機器人故障中。"

speech_config = speechsdk.SpeechConfig(subscription=config['AzureSpeech']['SPEECH_KEY'], 
                                       region=config['AzureSpeech']['SPEECH_REGION'])
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
UPLOAD_FOLDER = 'static'
def azure_speech(user_input):
    user_input=user_input.replace("*", "")
    if "\n\n\n翻譯結果:\n\n\n\n" in user_input:
        parts = user_input.split("\n\n\n翻譯結果:\n\n\n\n", maxsplit=1)  # 限制只分割一次
        #part1 = parts[0]
        #part2 = parts[1]
    # The language of the voice that speaks.
    #speech_config.speech_synthesis_voice_name = "ja-JP-NanamiNeural"
    speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoMultilingualNeural"
    file_name = "outputaudio.wav"
    #file_name1 = "outputaudio.wav"
    file_config = speechsdk.audio.AudioOutputConfig(filename="static/" + file_name)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=file_config
    )

    ssml_user_input = ""
    ssml_user_input += '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">'
    ssml_user_input += '<voice name="zh-CN-XiaoxiaoMultilingualNeural">'
    ssml_user_input += '<mstts:express-as style="Default" styledegree="2">'
    ssml_user_input += user_input
    ssml_user_input += "</mstts:express-as>"
    ssml_user_input += "</voice>"
    ssml_user_input += "</speak>"

    # Receives a text from console input and synthesizes it to wave file.
    #result = speech_synthesizer.speak_text_async(user_input).get()
    result = speech_synthesizer.speak_ssml_async(ssml_user_input).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(
            "Speech synthesized for text [{}], and the audio was saved to [{}]".format(
                user_input, file_name
            )
        )
        audio_duration = round(
            librosa.get_duration(path="static/outputaudio.wav") * 1000
        )
        print(audio_duration)
        return audio_duration
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
        return None


if __name__ == "__main__":
    app.run(port=5001)

