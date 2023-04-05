from pyllamacpp.model import Model

def new_text_callback(text: str):
    print(text, end="")

model = Model(ggml_model='..\gpt4all\chat\gpt4all-lora-converted.bin', n_ctx=512)
generated_text = model.generate("Once upon a time, ", n_predict=55)
print(generated_text)