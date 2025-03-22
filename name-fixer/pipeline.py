import re
from transformers import Text2TextGenerationPipeline

chars = "ÀÁÈÌÍÑÒÓÙÚìíñòú"
pattern = re.compile(rf"(?<=\w)\s+([{chars}])|([{chars}])\s+(?=\w)")

def fix_unknown_chars(text: str) -> str:
    "Fixes extra spacing added before and after the new chars added to the tokenizer. i.e Mu ñ oz -> Muñoz"
    return pattern.sub(lambda m: m.group(1) or m.group(2), text)


class NameFixerPipeline(Text2TextGenerationPipeline):
    def postprocess(self, model_outputs):
        outputs = super().postprocess(model_outputs)
        for o in outputs:
            text = fix_unknown_chars(o["generated_text"])
            parts = [p.strip() for p in text.split("|")]
            if len(parts) == 3:
                o["structured"] = {
                    "first_name": parts[0],
                    "last_name": parts[1],
                    "gender": parts[2]
                }
            else:
                o["structured"] = {"first_name": None, "last_name": None, "gender": None}
        return outputs