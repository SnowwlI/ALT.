import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Carregando o modelo GPT-Neo
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

def gerar_resposta(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta

# Função de exibição da conversa
def mostrar_conversa():
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    # Mostrar a conversa
    for mensagem in st.session_state.mensagens:
        if mensagem["role"] == "usuario":
            st.markdown(f"<div style='background-color: #E1F5FE; padding: 10px; border-radius: 10px;'>{mensagem['texto']}</div>", unsafe_allow_html=True)
        elif mensagem["role"] == "ia":
            st.markdown(f"<div style='background-color: #F1F8E9; padding: 10px; border-radius: 10px;'>{mensagem['texto']}</div>", unsafe_allow_html=True)

# Função para adicionar mensagem do usuário e IA
def adicionar_mensagem(role, texto):
    st.session_state.mensagens.append({"role": role, "texto": texto})

# Função principal
def main():
    st.title("Jarvis - Assistente de Tecnologia")

    # Entrada de texto do usuário
    prompt = st.text_input("Faça uma pergunta sobre tecnologia ou peças de computador:")

    # Exibir conversa
    mostrar_conversa()

    # Quando o usuário envia uma pergunta
    if prompt:
        # Adicionar a mensagem do usuário
        adicionar_mensagem("usuario", prompt)

        # Gerar a resposta da IA
        resposta = gerar_resposta(prompt)

        # Adicionar a resposta da IA
        adicionar_mensagem("ia", resposta)

        # Atualizar a conversa
        mostrar_conversa()

if __name__ == "__main__":
    main()
