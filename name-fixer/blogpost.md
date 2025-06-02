# LightWeight NLP Models Name Normalization with T5-Small

**TLDR**: I finetuned a lightweight deep learning model that effortlessly normalizes Spanish names: correcting accents, capitalization, clearly separating first names from surnames, and even predicting gender. And yes, it all happens in just three simple lines of Python:

```python
from transformers import pipeline

name_fixer = pipeline(model="juanluisrto/t5-small-namefixer-es", trust_remote_code=True)

name_fixer("Juan aantonio   rodriguez")
# [{'first_name': 'Juan Antonio', 'last_name': 'Rodríguez', 'gender': 'M'}]
```

## The problem:
Ever had to deal with a huge database filled with names inputed by users? I've lately been there and i can tell you it's a mess. You can find all sorts of things. Here you have a couple of examples of badly formatted names: 

| first_name       | last_name         |
|------------------|-------------------|
| ANAMARÍA         | GONZALEZ          |
| Jesus            | martinez Delgado  |
| dani             |  muñoz.           |
| SofiaRamirez     |                   |
| cristinaa        | fErnandez         |
| LUIS DE LA CRUZ  | DE LA CRUZ        |
|                  | patricia ruiz     |
| francisco jose   | delvalle          |
| Juan Carlos      | SÁNCHEZ           |
| Dr.              | Pedro Domínguez   |
| marina           |                   |
| GARCIA           | PEDRO MIGUEL      |
| MARTA LÓPEZ      | MARTA LÓPEZ       |
| Andrés           | Simón             |
| Almu             | aalvarado         |
| Juan     Pedro   | Campos            |
| angela           | villalobos        |

Improper accents, chaotic capitalization, unpredictable spacing, duplicated data, swapped fields... These issues are bearable if you are simply storing data internally, but they can become an obstacle if you need to enrich your users' profiles with third-party data sources. Then you might need to do name matching and if your data looks like this, it will be a nightmare.



## The solution:

To deal with this problem, I built a surprisingly efficient fix: a specialized T5-Small model, trained explicitly for Spanish names, capable of swiftly cleaning millions of records.

Here are some examples of inputs and outputs taken from the previous table:

| input                            | first_name      | last_name           | gender |
|----------------------------------|-----------------|---------------------|--------|
| ANAMARÍA GONZALEZ                | Ana María       | González            | F      |
| Jesus martinez Delgado           | Jesús           | Martínez Delgado    | M      |
| dani muñoz.                      | Dani            | Muñoz               | M      |
| SofiaRamirez                     | Sofía           | Ramírez             | F      |
| cristinaa fErnandez              | Cristina        | Fernández           | F      |
| LUIS DE LA CRUZ DE LA CRUZ      | Luis            | De la Cruz          | M      |
| patricia ruiz                    | Patricia        | Ruiz                | F      |
| francisco jose delvalle         | Francisco José  | Del Valle           | M      |
| Juan Carlos SÁNCHEZ              | Juan Carlos     | Sánchez             | M      |
| Dr. Pedro Domínguez              | Pedro           | Domínguez           | M      |
| marina                           | Marina          |                     | F      |
| GARCIA PEDRO MIGUEL             | Pedro Miguel    | García              | M      |
| MARTA LÓPEZ MARTA LÓPEZ         | Marta           | López               | F      |
| Andrés Simón                     | Andrés          | Simón               | M      |
| Almu aalvarado                   | Almu            | Alvarado            | F      |
| Juan     Pedro Campos            | Juan Pedro      | Campos              | M      |
| angela villalobos                | Ángela          | Villalobos          | F      |

It transforms messy inputs into neatly formatted outputs, restoring accents, separating names, and predicting gender—all within milliseconds, ideal if you have huge datasets to process.

## The Backstory: A Database in Chaos

The project began with a client's massive, chaotic database. Inconsistent name formats didn’t just make it hard to read: they hindered functionality, causing duplicate entries, faulty analytics, and broken search features.

A traditional solution like regex would've been cumbersome and brittle, failing to capture the nuances of Spanish naming conventions. Using LLMs would've been unfeasible due to the size of the dataset. Enter T5.

## Why T5-Small?

While the hype often surrounds newer, massive LLMs, T5 (Text-to-Text Transfer Transformer released by Google in 2019) quietly remains a powerful and efficient tool for specific text transformation tasks. At just 60 million parameters and 250Mb in weights, it's lean, fast, and perfectly suited for targeted tasks like name normalization.

## Crafting the Perfect Dataset

The success of the model hinged on a meticulously crafted training dataset:

- The starting point was a dataset found on the internet with 11 million spanish names. You don't really want to know where it came from 😈. 
- Generated millions of realistic "corruptions": removing accents, changing case, inserting irregular spaces, and scrambling surname orders. 
  
> _Example_: For the name "**Óscar Sánchez**" several realistic corruptions can occur: 
>   - removing accents and converting to lowercase => "oscar sanchez"; 
>   - inserting extra spaces or removing them => "Óscar  Sánchez ", "Óscarsanchez"
>   - applying random uppercase changes => "óScAr sÁnChez"; 
>   - duplicating letters => "Óscar sánchezz"
> These variations mimic common errors and transformations in real-world data, providing a robust training ground for the model to learn from.

The target? Restored names in the format: `"First Name|Last Name|Gender"`.

## Tackling T5’s Spanish Character Problem

An unexpected challenge arose: T5's tokenizer, built for English, struggled with Spanish-specific characters (ñ, á, é, í, etc.) which were not in the vocabulary. Instead of retraining the tokenizer, a significant undertaking, I strategically added these characters as special tokens.

However, this introduced another quirk: T5 placed unwanted spaces around these special characters (e.g., "Mu ñ oz"). To address this swiftly, I implemented a simple yet effective regex-based post-processing step:

```python
chars = "ÀÁÈÌÍÑÒÓÙÚìíñòú"
pattern = re.compile(rf"(?<=\w)\s+([{chars}])|([{chars}])\s+(?=\w)")

def fix_unknown_chars(text: str) -> str:
    """Removes unwanted spaces around special characters."""
    return pattern.sub(lambda m: m.group(1) or m.group(2), text)
```


## Training the Model

To train our `t5-small` based model, I followed several steps.

I began by initializing the T5 tokenizer and model. Then I added the special unknown characters to the tokenizer and resized the model's token embeddings. This is necessary because since the model is not able to recognize these characters, it will treat all of them as the same <unk> (unknown) token. We want to at least let the model know that these characters are different.

```python
import multiprocessing
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Initialize the T5 tokenizer and model (T5-small in this case)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


tokenizer.add_tokens(set(chars))
model.resize_token_embeddings(len(tokenizer))
```

Thereafter, I loaded the previously created dataset from the Hugging Face Hub and tokenized it with our custom tokenizer.

```python
from datasets import load_dataset

# Load dataset from Hugging Face Hub
hf_dataset = load_dataset("juanluisrto/names-es")

num_cpus = multiprocessing.cpu_count()
print(f"{num_cpus} CPUs available")

def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=64, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the datasets
tokenized_datasets = hf_dataset.map(preprocess_function, batched=True, num_proc=num_cpus)
```

The dataset was tokenized using a preprocessing function that ensures both inputs and outputs are appropriately formatted for the model.
We set the maximum length of the input and output to 64 tokens because names are typically short strings, so longer sequences are unnecessary. This not only conserves memory but also speeds up training and inference, as the model processes fewer tokens per example.

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="steps",
    save_steps=1500,        # Save a checkpoint every 1500 steps
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    report_to=["tensorboard"],
    push_to_hub=True,              # Enable pushing to Hugging Face Hub after training
    hub_model_id="juanluisrto/t5-small-namefixer-es",
)
```

The training configuration was set up with specific hyperparameters, including a learning rate of \(2 \times 10^{-5}\), batch sizes of 8 for both training and evaluation, and a total of 3 epochs. With these settings, I aimed to optimize the model's learning process while preventing overfitting.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()
```

The `Trainer` class from the Hugging Face Transformers library was used to manage the training loop, including evaluation and logging. This streamlined the process and allowed for easy integration with tools like TensorBoard for monitoring.

### Evaluation Metrics

The model, `t5-small-namefixer-es`, achieved an evaluation loss of 0.0064, indicating a high level of accuracy in its predictions. The evaluation runtime was 2.8981 seconds, with a throughput of 345.057 samples per second and 43.132 steps per second, demonstrating efficient processing capabilities.

For more detailed insights into the training process, you can view the [TensorBoard dashboard](https://huggingface.co/juanluisrto/t5-small-namefixer-es/tensorboard).


## Building an Intuitive Pipeline

As shown in the beginning of the post, I encapsulated the entire solution within a streamlined Hugging Face pipeline. Below you have the implementation.

```python
class NameFixerPipeline(Text2TextGenerationPipeline):
    def postprocess(self, model_outputs):
        outputs = super().postprocess(model_outputs)
        results = []
        for output in outputs:
            text = fix_unknown_chars(output["generated_text"])
            first_name, last_name, gender = (text.split("|") + [None, None, None])[:3]
            results.append({
                "first_name": first_name.strip() or None,
                "last_name": last_name.strip() or None,
                "gender": gender.strip() or None
            })
        return results
```

With this pipeline, anyone can effortlessly obtain neatly structured outputs without worrying about underlying complexities.


## Results: Efficiency Meets Accuracy

The fine-tuned T5-Small model effectively addressed the client’s needs:

- **Speed**: Consistently processed hundreds of names per second on modest hardware.
- **Impact**: Millions of records cleaned at low cost within hours instead of weeks
- **Accuracy**: Significantly reduced data inconsistencies, enabling accurate analytics and efficient duplicate detection.

My client’s database was revitalized, dramatically improving duplicate detection and reliability of analytics.

## Key Takeaway: Matching the Model to the Task

This experience reinforced the value of carefully matching NLP models to specific tasks. For Spanish name normalization, a specialized model like T5-Small offered a practical solution—lightweight, fast, and sufficiently accurate. While larger, newer models may excel in versatility, targeted models often provide the best balance of performance, speed, and deployment simplicity.

Explore the model and code yourself on Hugging Face and GitHub:

- **Model**: `juanluisrto/t5-small-namefixer-es`
- **Code**: [GitHub Repository](https://github.com/juanluisrto/blogposts/tree/main/name-fixer)

Next time you're solving a specific NLP challenge, consider a specialized model like T5. Sometimes, the older models still have the best tricks up their sleeves.

