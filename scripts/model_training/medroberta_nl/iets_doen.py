from transformers import pipeline, Conversation


chatbot = pipeline(task='conversational', model='Rijgersberg/GEITje-7B-chat-v2',
                   device_map='auto')

print(chatbot(
    Conversation("""Hoe kun je muziek genereren met behulp van Large Language Models?""")
))