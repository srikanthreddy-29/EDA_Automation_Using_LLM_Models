import gradio as gr 
import ollama
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_analysis(file_path):
    df = pd.read_csv(file_path)

    for col in df.select_dtypes(include = ['number']).columns:
        df[col].fillna(df[col].median())


    for col in df.select_dtypes(include = ['object']).columns:
        df[col].fillna(df[col].mode()[0])
    
    summary = df.describe(include="all").to_string()

    missing_values = df.isnull().sum().to_string()

    insights = generate_insights(summary)

    plot_paths = generate_visualizations(df)

    return f"\n Data loaded Successfully \n\n Summary:\n{summary}\n\n Missing values:\n {missing_values} \n\n LLM Insights:\n {insights}\n\n plots:\n{plot_paths}"

def generate_insights(df_summary):
    prompt = f"Analyse the dataset insights and provide insights:\n\n{df_summary}"
    response = ollama.chat(model="mistral",messages = [{"role":"user","content":prompt}])
    return response['choices'][0]['message']['content']

    
def generate_visualizations(df):
    plot_paths = []

    for col in df.select_dtypes(include = ['number']).columns:
        plt.figure(figsize=(7,5))
        sns.histplot(df[col],bins = 30, kde=True,color='blue')
        plt.title(f"Distribution of {col}")
        path = f"{col}_distribution.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()

    # corelation Heatmap
    numeric_df = df.select_dtypes(include = ['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(8,5))
        sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',fmt='2f',linewidths=0.5)
        plt.title("correlation Heatmap")
        path = "correlation_heatmap.png"
        plt.savefig(path)
        plot_paths.append(path)
        plt.close()


    
    return plot_paths


app = gr.Interface(fn = eda_analysis,
                   inputs= gr.File(type = "filepath"),
                   outputs=[gr.Textbox(label= "EDA REPORT"),gr.File(label="Data visualization")],
                   title="LLM POWERED EDA DATA ANLYZER APP",
                   description= "upload any dataset to view the EDA report and data visualizations")

app.launch(share=True)
                    
