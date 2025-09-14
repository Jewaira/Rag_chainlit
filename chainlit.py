import chainlit as cl
import httpx

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="🤖 Hello! Ask me about cats or scientific experiments!").send()

@cl.on_message  
async def main(message: cl.Message):
    history = cl.user_session.get("history", [])

    # Build payload
    payload = {
        "query": message.content,
        "history": history
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            cl.user_session.set("history", data.get("history", []))

            await cl.Message(content=data["response"]).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()
