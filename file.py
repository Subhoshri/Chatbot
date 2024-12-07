import openai

#OpenAI API key
openai.api_key="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"

def chat_with_gpt(user):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  #Choose a Model
        messages=[{"role":"user","content":user}]
        
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    #Take user input
    while True:
        user_msg = input("You: ")
        if(user_msg.lower() in ["exit","quit","end","bye"]):
            break
        bot_msg = chat_with_gpt(user_msg)

        print("Chatbot: ", bot_msg)
