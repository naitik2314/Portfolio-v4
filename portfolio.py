import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import precision_recall_curve

# Page Configuration
st.set_page_config(page_title="Naitik SHah", page_icon=":guardsman:", layout="wide")

# Header Section
st.title("💻 Naitik Shah")
st.markdown("""
Welcome to my **Data Science** portfolio! With 2+ years of hands-on experience in machine learning, predictive modeling, and data engineering,  
I specialize in transforming data into actionable insights. My toolset includes Python, SQL, Databricks, Spark, and AWS.  
Explore my work, visualizations, and projects as I showcase how I leverage data science to drive business solutions.
""")

# Interactive Skill Showcase (Hover Effects)
st.header("🔧 Key Skills")
skills = ['Python', 'SQL', 'Machine Learning', 'TensorFlow', 'Spark', 'AWS', 'ETL', 'Data Visualization']
st.markdown("Hover over the skills to see more details about each!")
for skill in skills:
    st.markdown(f"<span style='color:#3498db; font-size:16px;'>{skill}</span>", unsafe_allow_html=True)

# Dynamic Visualizations - Show a colorful heatmap for your skills
st.header("🌍 Skills Heatmap")
heatmap_data = pd.DataFrame({
    'Skills': skills,
    'Proficiency': [80, 85, 95, 90, 88, 92, 87, 85],
    'Usage': [90, 85, 70, 80, 85, 90, 88, 75]
})
fig = px.scatter(heatmap_data, x='Skills', y='Proficiency', size='Usage', color='Skills', 
                 title="Skills Proficiency and Usage", color_continuous_scale='Viridis')
st.plotly_chart(fig)

# Work Experience Section - Full Time Jobs Focus with Interactive Metrics
st.header("🚀 Work Experience")
st.subheader("Data Scientist Co-op at DTE Energy (Jan 2024 - Dec 2024)")

st.markdown("""
At **DTE Energy**, I worked as a **Data Scientist Co-op** where I focused on **applying machine learning** and **data engineering** techniques to drive business value.  
I collaborated with cross-functional teams to optimize operations, reduce costs, and improve customer engagement.

Key achievements:
""")

# Using Bar Chart to Show Impact of Predictive Modeling
predictive_modeling_data = {
    'Metric': ['Risk Identification Accuracy', 'Operational Efficiency Improvement', 'Fraud Reduction'],
    'Value': [75, 10, 15]  # 75% accuracy, 10% improvement, 15% reduction
}

df_predictive_modeling = pd.DataFrame(predictive_modeling_data)

fig_predictive = px.bar(df_predictive_modeling, x='Metric', y='Value', color='Metric', 
                        title="Impact of Data Science Projects at DTE Energy",
                        labels={'Value': 'Percentage (%)', 'Metric': 'Key Metric'})
st.plotly_chart(fig_predictive)

st.markdown("""
- **Predictive Modeling**: I built **predictive models** using **LightGBM** and **TensorFlow** to **identify high-risk customers** with **75% accuracy**.
- **A/B Testing**: Led A/B testing initiatives using **Python**, **SQL**, and **Power BI**. This resulted in a **10% reduction** in call handling time.
- **Fraud Detection**: I used **regression analysis** and **clustering techniques** to identify fraudulent customer returns, resulting in a **15% reduction** in financial losses.
""")

# Add the horizontal rule inside markdown
st.markdown("---")

st.subheader("Data Analyst at Nylex Group (Oct 2018 - Dec 2019)")

st.markdown("""
At **Nylex Group**, I worked as a **Data Analyst**, where I played a key role in **optimizing data pipelines** and **improving reporting efficiency** across manufacturing operations.

Key achievements:
""")

# Using Bar Chart to Show Key Impact Metrics in Data Analytics
data_analytics_data = {
    'Metric': ['Data Processing Efficiency', 'Automation Efficiency'],
    'Value': [15, 20]  # 15% improvement, 20% efficiency gain
}

df_data_analytics = pd.DataFrame(data_analytics_data)

fig_data_analytics = px.bar(df_data_analytics, x='Metric', y='Value', color='Metric', 
                            title="Impact of Data Analytics at Nylex Group", 
                            labels={'Value': 'Percentage (%)', 'Metric': 'Key Metric'})
st.plotly_chart(fig_data_analytics)

st.markdown("""
- **Data Pipeline Optimization**: Improved **data processing efficiency** by **15%** using **SQL optimization** and **data wrangling**.
- **Real-Time Automation**: Automated real-time data analysis with **Python**, **Numpy**, and **APIs**, resulting in a **20% increase in efficiency**.
- **KPI Reporting**: Designed and developed **KPIs** using **Power BI** and **SQL**, improving decision-making by **10%**.
""")

# Education Section with Interactive Academic Achievements
st.header("🎓 Academic Achievements")
st.markdown("""
I hold a **Master's in Computer Science** from **Wayne State University** with a **4.0 GPA**, and a **Bachelor's in Computer Engineering** from **Mumbai University**. I have consistently demonstrated excellence in both my academic and professional work, especially in areas like machine learning, data analytics, and data engineering.
""")

# GPA Bar Chart for Academic Excellence
gpa_data = {
    "Institution": ["Wayne State University", "Mumbai University"],
    "Degree": ["M.S. in Computer Science", "B.E. in Computer Engineering"],
    "GPA": [4.0, 3.43],
    "Year": [2024, 2022]
}

df_gpa = pd.DataFrame(gpa_data)

fig_gpa = px.bar(df_gpa, x="Institution", y="GPA", color="Degree", title="GPA and Academic Achievements",
                 labels={"GPA": "GPA", "Institution": "Institution"})
st.plotly_chart(fig_gpa)

# Detailed Educational Background (Interactive Component)
st.markdown("""
### Way of Learning
At **Wayne State University**, I focused on:
- **Machine Learning**
- **Data Engineering**
- **Predictive Analytics**

### Key Courses:
- **Advanced Machine Learning**
- **Data Structures & Algorithms**
- **Cloud Computing**
  
At **Mumbai University**, I studied:
- **Data Science** 
- **Cloud Computing**
""")

# Projects Section with Data Science Workflows
st.header("🚀 Projects & Achievements")

# Interactive Project Details (Collapsible Sections)
with st.expander("Energy Anomaly Detection (Oct 2024)"):
    st.markdown("""
    **Problem:** Detect anomalies in real-time energy consumption to optimize energy resource allocation.  
    **Solution:** I built a real-time anomaly detection pipeline using **AWS services** such as **S3**, **Lambda**, and **SageMaker**. The model used **LSTM** networks for time-series forecasting and achieved **85% accuracy** in detecting anomalies.
    
    **Steps Taken:**
    - Data collection from sensors and energy usage APIs.
    - Data cleaning using **PySpark** and integration into **AWS S3**.
    - **LSTM** model training in **SageMaker** and deployment via **EC2 instances**.
    """)
    st.write("Check out the code and results [GitHub Repo](https://github.com)")
    
with st.expander("Credit Risk Scoring Model (Sep 2024)"):
    st.markdown("""
    **Problem:** Predict loan defaults with high accuracy to assist financial institutions.  
    **Solution:** Developed a **Logistic Regression** and **XGBoost** model, achieving **82% accuracy** using **AWS** for model storage and deployment.  
    This project showcases how I build scalable models for real-world applications.
    
    **Steps Taken:**
    - Data preprocessing using **Pandas** and **SQL**.
    - Built and tuned a model using **XGBoost** and **Logistic Regression**.
    - Deployed the model on **AWS EC2** for real-time predictions.
    """)
    st.write("See more details on this project [GitHub Repo](https://github.com)")

# Interactive Model Evaluation - Precision-Recall Curve
st.header("📊 Model Performance - Precision-Recall Curve")
st.markdown("""
Model performance is essential in every data science project. Below is an interactive **Precision-Recall Curve** showing the performance of one of my models.
""")

# Generate synthetic data for Precision-Recall Curve
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0])  # Example true labels
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.3, 0.7, 0.5, 0.85, 0.9, 0.2])  # Example prediction scores

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Plotting the Precision-Recall curve
fig = go.Figure(data=[
    go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve')
])
fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
st.plotly_chart(fig)

# Contact Section (Optional)
st.header("📬 Contact")
st.markdown("""
Feel free to reach out to me via [LinkedIn](https://www.linkedin.com/in/naitik-shah-49baba1a1/) or email at naitik@wayne.edu.  
Let's discuss how we can collaborate to turn data into decisions!
""")

# Footer with Links
st.markdown("""
© 2024 Naitik Shah. All rights reserved. | [GitHub](https://github.com) | [LinkedIn](https://www.linkedin.com/in/naitik-shah-49baba1a1/)
""")
