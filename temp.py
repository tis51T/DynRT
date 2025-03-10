from transformers import RobertaModel, RobertaTokenizer

# Tên mô hình (có thể thay đổi thành "roberta-large" nếu cần)
model_name = "roberta-base"

# Tải mô hình RoBERTa
model = RobertaModel.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Lưu mô hình và tokenizer vào thư mục
save_directory = "./roberta-base/"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"RoBERTa đã được lưu vào {save_directory}")
