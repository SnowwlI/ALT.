import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Carregando modelo e tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    return tokenizer, model

tokenizer, model = load_model()

st.title("🤖 GPT-J - Assistente Técnico")

# Histórico de mensagens
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Digite sua pergunta ou prompt técnico:")

temperature = st.sidebar.slider("🎨 Criatividade (temperature)", 0.1, 1.0, 0.3, 0.1)
max_length = st.sidebar.slider("📏 Comprimento da resposta (tokens)", 50, 512, 200, 10)

# Mostrar histórico
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Quando usuário envia nova mensagem
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando... 🧠"):
            # Adiciona um prefixo técnico
            prompt = f"""Você é um assistente técnico especializado. Responda de forma formal, objetiva e baseada em dados.

Pergunta: {user_input}
Resposta:"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            resposta = generated_text.split("Resposta:")[-1].strip()

        st.markdown(resposta)
        st.session_state.chat_history.append(("assistant", resposta))
