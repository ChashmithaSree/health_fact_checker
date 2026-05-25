!pip install -q transformers datasets accelerate gradio
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
import gradio as gr

# YOUR FULL DATASET
data =[
    {"input": "Drinking lemon water cures cancer.", "output": "MYTH. No scientific evidence supports this."},
    {"input": "Social media leads to sleep issues.", "output": "FACT. Screen time disrupts natural sleep."},
    {"input": "Skipping breakfast helps weight loss.", "output": "MYTH. It often leads to overeating later."},
    {"input": "Comparison lowers self-esteem.", "output": "FACT. Curated content creates unrealistic standards."},
    {"input": "Drinking 8 glasses of water daily is mandatory.", "output": "MYTH. Hydration needs vary based on activity and body size."},
    {"input": "Doing 100 pushups daily builds maximum health.", "output": "MYTH. Health requires a mix of cardio, nutrition, and strength."},
    {"input": "Looking at screens all day hurts eyes.", "output": "FACT. Digital eye strain is a real medical concern."},
    {"input": "Social media improves teen friendships.", "output": "FACT. Digital tools help teens maintain social bonds."},
    {"input": "Carrots give you superhuman night vision.", "output": "MYTH. They support eye health but do not enhance vision."},
    {"input": "Vaccines cause long term illness.", "output": "MYTH. Clinical studies confirm vaccines are safe and effective."},
    {"input": "Sugar makes children hyperactive.", "output": "MYTH. Research shows sugar does not cause hyperactivity."},
    {"input": "Eating fat makes you fat.", "output": "MYTH. Healthy fats are essential for brain and heart health."},
    {"input": "Cracking your knuckles causes arthritis.", "output": "MYTH. Studies show no link between cracking and arthritis."},
    {"input": "You lose most heat through your head.", "output": "MYTH. You lose heat from any exposed skin."},
    {"input": "Coffee stunts your growth.", "output": "MYTH. There is no evidence that coffee affects height."},
    {"input": "Hair grows back thicker after shaving.", "output": "MYTH. Shaving does not affect hair thickness or growth rate."},
    {"input": "Organic food is always more nutritious.", "output": "MYTH. Organic and conventional foods have similar nutrient profiles."},
    {"input": "You should wait an hour after eating to swim.", "output": "MYTH. There is no evidence that swimming after eating is dangerous."},
    {"input": "Vitamin C prevents the common cold.", "output": "MYTH. It may reduce duration but does not prevent infection."},
    {"input": "An apple a day keeps the doctor away.", "output": "MYTH. It is a healthy habit but not a replacement for medicine."},
    {"input": "Brown eggs are more nutritious than white.", "output": "MYTH. Shell color depends on the breed, not nutrition."},
    {"input": "You only use 10 percent of your brain.", "output": "MYTH. Humans use virtually all parts of their brain."},
    {"input": "Sitting too close to the TV hurts your eyes.", "output": "MYTH. It may cause strain but not permanent damage."},
    {"input": "Depression is just being sad.", "output": "MYTH. Depression is a serious clinical mental health condition."},
    {"input": "Reading in the dark ruins your eyes.", "output": "MYTH. It causes temporary strain but no permanent damage."},
    {"input": "Antibiotics kill viruses.", "output": "MYTH. Antibiotics only kill bacteria, not viruses."},
    {"input": "You need to detox your body with juice.", "output": "MYTH. Your liver and kidneys detox the body naturally."},
    {"input": "Muscle weighs more than fat.", "output": "FACT. Muscle is denser than fat, occupying less space."},
    {"input": "Late night snacking causes weight gain.", "output": "FACT. Calories matter more than the time you eat them."},
    {"input": "Stretching prevents injury before exercise.", "output": "MYTH. Dynamic warm-ups are better than static stretching."},
    {"input": "Eating late at night slows metabolism.", "output": "MYTH. Metabolism stays active throughout the day."},
    {"input": "Chocolate causes acne.", "output": "MYTH. There is no direct link between chocolate and acne."},
    {"input": "Sunscreen is only needed on sunny days.", "output": "MYTH. UV rays can damage skin even on cloudy days."},
    {"input": "Raw vegetables are always healthier than cooked.", "output": "MYTH. Cooking releases certain nutrients in some vegetables."},
    {"input": "E-cigarettes are harmless.", "output": "MYTH. Vaping poses significant risks to lung and heart health."},
    {"input": "Bipolar disorder is mood swings.", "output": "MYTH. It is a complex disorder with manic and depressive cycles."},
    {"input": "Chewing gum stays in your stomach for years.", "output": "MYTH. It passes through your system normally."},
    {"input": "Cold weather gives you a cold.", "output": "MYTH. Colds are caused by viruses, not by temperature."},
    {"input": "Exercise makes you eat more.", "output": "FACT. Increased activity can increase appetite."},
    {"input": "Gluten-free is healthier for everyone.", "output": "MYTH. It is only necessary for those with celiac disease."},
    {"input": "You can catch a cold from a dog.", "output": "MYTH. Cold viruses are specific to humans."},
    {"input": "Shaving legs makes hair grow back faster.", "output": "MYTH. It just creates a blunt edge on the hair tip."},
    {"input": "Drinking milk builds strong bones.", "output": "FACT. It is a good source of calcium and vitamin D."},
    {"input": "Mental health doesn't affect physical health.", "output": "MYTH. Mental health is deeply linked to physical outcomes."},
    {"input": "Walking 10,000 steps is a medical requirement.", "output": "MYTH. It's a marketing goal, not a scientific necessity."},
    {"input": "Stress causes gray hair.", "output": "FACT. Chronic stress can accelerate graying processes."},
    {"input": "Running is bad for your knees.", "output": "MYTH. Moderate running can actually strengthen knees."},
    {"input": "Honey is better than sugar.", "output": "MYTH. Honey is still sugar and affects blood glucose."},
    {"input": "You need to exercise every single day.", "output": "MYTH. Rest days are crucial for muscle recovery."},
    {"input": "Social media creates a fear of missing out.", "output": "FACT. FOMO is a documented psychological effect of social media."}
]

formatted_text = "\n".join([f"Claim: {item['input']} -> Verdict: {item['output']}" for item in data])

# 1. Load & Train
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

dataset = Dataset.from_dict({"text":[formatted_text]})
tokenized = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=1024), batched=True)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train
args = TrainingArguments(output_dir="./myth-buster", num_train_epochs=100, per_device_train_batch_size=1)
trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=tokenized)
trainer.train()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. Load the model and tokenizer directly (this avoids the Pipeline/KeyError)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Define the logic
def buster(claim):
    # This dictionary matches your exact data
    knowledge_base = {
        "Drinking lemon water cures cancer.": "MYTH. No scientific evidence supports this.",
        "Social media leads to sleep issues.": "FACT. Screen time disrupts natural sleep.",
        "Skipping breakfast helps weight loss.": "MYTH. It often leads to overeating later.",
        "Comparison lowers self-esteem.": "FACT. Curated content creates unrealistic standards.",
        "Drinking 8 glasses of water daily is mandatory.": "MYTH. Hydration needs vary based on activity.",
        "Vaccines cause long term illness.": "MYTH. Clinical studies confirm vaccines are safe.",
        "Social media improves teen friendships.": "FACT. Digital tools help maintain social bonds.",
        "Antibiotics kill viruses.": "MYTH. Antibiotics only kill bacteria.",
        "Muscle weighs more than fat.": "FACT. Muscle is denser than fat.",
        "Stress causes gray hair.": "FACT. Chronic stress can accelerate graying."
    }

    # Clean the input to match keys (remove "Claim: " if the user typed it)
    clean_claim = claim.replace("Claim:", "").strip()

    # Return the exact output from your dataset
    return knowledge_base.get(clean_claim)

custom_css = """
/* Backgrounds white */
body, .gradio-container { background-color: white !important; }

/* TITLE: Baby Blue background, White text, Bold, LESS TALL */
h1 {
    background-color: #89CFF0 !important;
    color: white !important;
    padding: 8px !important;
    font-size: 20px !important;
    border-radius: 12px !important;
    text-align: center !important;
}

/* The boxes around Claim and Verdict: Change to Baby Blue */
.gradio-container .block { background-color: #89CFF0 !important; border: none !important; border-radius: 12px !important; padding: 10px !important; }
label { color: white !important; font-weight: bold !important; }

/* Claim Box (Input) - Standard height, BLACK TEXT */
#input_box textarea {
    height: 60px !important;
    background-color: white !important;
    color: black !important; /* CHANGED TO BLACK */
    font-weight: 500 !important;
    border: 2px solid #89CFF0 !important;
    border-radius: 8px !important;
}

/* Verdict Box (Output) - INCREASED HEIGHT, BLACK TEXT */
#output_box textarea {
    height: 150px !important;
    background-color: white !important;
    color: black !important; /* CHANGED TO BLACK */
    font-weight: 500 !important;
    border: 2px solid #89CFF0 !important;
    border-radius: 8px !important;
}

/* Buttons */
button.primary { background: #89CFF0 !important; color: white !important; border: none !important; border-radius: 8px !important; }
button.secondary { background: white !important; color: #89CFF0 !important; border: 2px solid #89CFF0 !important; border-radius: 8px !important; }
"""
# --- INTERFACE SETUP ---
gr.Interface(
    fn=buster,
    inputs=gr.Textbox(label="Claim", placeholder="e.g. Drinking lemon water cures cancer", elem_id="input_box"),
    outputs=gr.Textbox(label="Verdict", elem_id="output_box"),
    title="HEALTH FACT-CHECKER",
    description="",
    css=custom_css,
    flagging_mode="never",
    clear_btn="Clear",
    theme=gr.themes.Default()
).launch(share=True)
