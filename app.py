import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from io import StringIO
matplotlib.use("Agg")

# --------------------------------------------------
# Page config with better styling
# --------------------------------------------------
st.set_page_config(
    page_title="Human Gut AMR Burden Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D5D5D;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #F0F7FF;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 20px 0;
    }
    .result-box {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        margin: 15px 0;
    }
    .risk-low { color: #28A745; font-weight: bold; }
    .risk-moderate { color: #FFC107; font-weight: bold; }
    .risk-high { color: #DC3545; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown('<h1 class="main-header">🧬 Human Gut AMR Burden Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine learning-powered assessment of antimicrobial resistance burden in human gut microbiome samples</p>', unsafe_allow_html=True)

# --------------------------------------------------
# Information Section
# --------------------------------------------------
with st.expander("📋 About This Tool", expanded=False):
    st.markdown("""
    ### What is AMR Burden Prediction?
    
    Antimicrobial Resistance (AMR) in the gut microbiome possess significant health risks. This tool uses machine learning to predict the AMR burden based on the abundance of resistance genes in human gut samples.
    
    ### How It Works:
    1. **Upload Data**: Provide a CSV file with AMR gene abundance data
    2. **Analysis**: Our model analyzes 50 key AMR genes identified through SHAP analysis
    3. **Prediction**: Calculates an AMR burden score and categorizes risk level
    4. **Visualization**: Provides detailed mechanism profiles and interpretations
    
    ### Data Requirements:
    - CSV format with samples as rows and AMR genes as columns
    - Must include all 50 key AMR genes used in the model
    - Gene abundance values should be normalized counts
    
    ### Applications:
    - Clinical risk assessment
    - Research studies on microbiome resistance
    - Public health surveillance
    """)

# --------------------------------------------------
# Sidebar with additional info
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🛠️ How to Use")
    st.info("""
    1. Prepare your CSV file with gene abundance data
    2. Ensure it includes the required 50 AMR genes
    3. Upload using the file uploader below
    4. View results for each sample
    """)
    
    st.markdown("## 📊 Model Information")
    st.markdown("**Model Type:** Huber Regressor")
    st.markdown("**Features:** 50 key AMR genes")
    st.markdown("**Scaler:** StandardScaler (trained on top 50 genes)")
    
    st.markdown("## ⚠️ Important Notes")
    st.warning("""
    - This tool is for research purposes
    - Results should be interpreted by qualified professionals
    - Always validate predictions with clinical data
    """)

# --------------------------------------------------
# Paths
# --------------------------------------------------
MODEL_PATH = "huber_amr_model.pkl"
SCALER_PATH = "scaler_top50.pkl"
GENE_INFO_PATH = "top50_shap_genes_annotated.csv"

# --------------------------------------------------
# Load model & metadata
# --------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        gene_info = pd.read_csv(GENE_INFO_PATH)
        
        # Exact pattern replacement for the specific format in your image
        # First, let's standardize all variations to "Non-Specific Resistance"
        
        # Create a mapping of patterns to replace
        replace_dict = {
            "Other / Unclassified": "Non-Specific Resistance",
            "Other/Unclassified": "Non-Specific Resistance", 
            "Other /Unclassified": "Non-Specific Resistance",
            "Other/ Unclassified": "Non-Specific Resistance",
            "general/Unknown": "Non-Specific Resistance",
            "general / Unknown": "Non-Specific Resistance",
            "Other": "Non-Specific Resistance",
            "Unclassified": "Non-Specific Resistance",
            "Unknown": "Non-Specific Resistance",
            "Non-specific": "Non-Specific Resistance",
            "non-specific": "Non-Specific Resistance"
        }
        
        # Apply the replacement
        gene_info["Resistance_Mechanism"] = gene_info["Resistance_Mechanism"].replace(replace_dict)
        
        # Also do a case-insensitive check for any remaining variations
        def final_cleanup(mechanism):
            mech = str(mechanism).strip()
            # Check if it contains any of our target words (case insensitive)
            if any(word in mech.lower() for word in ["other", "unclassified", "unknown", "non-specific", "general"]):
                return "Non-Specific Resistance"
            return mech
        
        gene_info["Resistance_Mechanism"] = gene_info["Resistance_Mechanism"].apply(final_cleanup)
        
        return model, scaler, gene_info
    except Exception as e:
        st.error(f"Error loading model assets: {str(e)}")
        return None, None, None

model, scaler, gene_info = load_assets()

if model is None:
    st.stop()

TOP_GENES = gene_info["AMR_Gene"].tolist()
GENE_TO_MECH = dict(zip(
    gene_info["AMR_Gene"],
    gene_info["Resistance_Mechanism"]
))

# --------------------------------------------------
# Helper functions - FIXED INTERPRET FUNCTION
# --------------------------------------------------
def risk_category(score):
    if score < 3e6:
        return "Low"
    elif score < 5e6:
        return "Moderate"
    return "High"

def get_risk_color(category):
    if category == "Low":
        return "risk-low"
    elif category == "Moderate":
        return "risk-moderate"
    return "risk-high"

def compute_sample_mechanisms(sample):
    mech_scores = {}
    for gene, value in sample.items():
        mech = GENE_TO_MECH.get(gene, "Non-Specific Resistance")
        mech_scores[mech] = mech_scores.get(mech, 0) + value
    
    total = sum(mech_scores.values())
    if total > 0:
        return {k: round(v / total, 3) for k, v in mech_scores.items()}
    return {k: 0 for k in mech_scores}

def interpret(mech_profile):
    if not mech_profile:
        return "No significant resistance mechanisms detected"
    
    dominant = max(mech_profile, key=mech_profile.get)
    dominant_percentage = mech_profile[dominant]
    
    # Debug: Show what we're working with
    # st.write(f"DEBUG: Dominant mechanism = '{dominant}' ({dominant_percentage:.1%})")
    
    if dominant_percentage < 0.3:
        return "Multiple resistance mechanisms contributing equally"
    
    # Check for Non-Specific Resistance - FIXED
    # Convert to lowercase for case-insensitive comparison
    dominant_lower = dominant.lower()
    
    if "non-specific" in dominant_lower:
        return "Resistance is dominated by Non-Specific Resistance mechanisms with indirect AMR contribution"
    elif "β-lactamase" in dominant or "beta-lactamase" in dominant_lower:
        return "β-lactamase–mediated resistance is prominent"
    elif "efflux" in dominant_lower:
        return "Efflux-based multidrug resistance likely"
    else:
        return "Mixed resistance mechanisms observed"

# --------------------------------------------------
# File upload section
# --------------------------------------------------
st.markdown("## 📤 Upload Your Data")
st.markdown("""
<div class="info-box">
    <strong>File Format:</strong> CSV with samples as rows and AMR genes as columns<br>
    <strong>Required Genes:</strong> Must include all 50 key AMR genes used in the model<br>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "**Drag and drop your CSV file here**",
    type=["csv"],
    help="Upload gene abundance CSV (rows = samples, columns = AMR genes)"
)

# Show sample data structure
with st.expander("📁 Expected Data Structure", expanded=True):
    st.markdown("""
    Your CSV should have this structure:
    
    | Sample_ID | gene_1 | gene_2 | ... | gene_50 |
    |-----------|--------|--------|-----|--------|
    | Sample_1  | 0.123  | 0.456  | ... | 0.789  |
    | Sample_2  | 0.234  | 0.567  | ... | 0.890  |
    
    **Note:** The first column should be Sample_ID
    """)

# --------------------------------------------------
# Process uploaded file
# --------------------------------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, index_col=0)
        
        # Check for required genes
        missing = set(TOP_GENES) - set(df.columns)
        if missing:
            st.error(f"❌ Missing {len(missing)} required genes")
            st.info(f"First 5 missing genes: {list(missing)[:5]}")
            
            # Show available vs required genes
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Required Genes (50 total):**")
                st.write(f"Found: {len(set(TOP_GENES) & set(df.columns))}")
                st.write(f"Missing: {len(missing)}")
            
            with col2:
                st.download_button(
                    "📥 Download Gene List",
                    data="\n".join(TOP_GENES),
                    file_name="required_genes.txt",
                    mime="text/plain"
                )
            st.stop()
        
        # Prepare data
        df = df[TOP_GENES]
        X_scaled = scaler.transform(df)
        scores = model.predict(X_scaled)
        
        # Summary statistics
        st.success(f"✅ Successfully analyzed {len(df)} samples")
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples Analyzed", len(df))
        with col2:
            avg_score = np.mean(scores)
            st.metric("Average AMR Score", f"{avg_score:,.0f}")
        with col3:
            high_risk = sum(1 for s in scores if risk_category(s) == "High")
            st.metric("High Risk Samples", high_risk)
        
        # --------------------------------------------------
        # Per-sample output with better UI
        # --------------------------------------------------
        st.markdown("## 📊 Results by Sample")
        
        for i, sample_id in enumerate(df.index):
            sample_score = float(scores[i])
            risk_cat = risk_category(sample_score)
            risk_color = get_risk_color(risk_cat)
            
            # Calculate mechanism profile
            mech_profile = compute_sample_mechanisms(df.loc[sample_id])
            
            # Create a clean results display
            with st.container():
                st.markdown(f"### 🧪 Sample: `{sample_id}`")
                
                # Metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**AMR Burden Score**")
                    st.markdown(f"<h2 style='margin-top:0;'>{sample_score:,.2f}</h2>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**Risk Category**")
                    st.markdown(f"<h3 class='{risk_color}'>{risk_cat}</h3>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"**Interpretation**")
                    st.info(interpret(mech_profile))
                
                # Resistance Mechanism Profile
                st.markdown("##### Resistance Mechanism Profile")
                
                # Create a better visualization
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    # Mechanism table
                    mech_df = pd.DataFrame({
                        'Mechanism': list(mech_profile.keys()),
                        'Proportion': list(mech_profile.values())
                    }).sort_values('Proportion', ascending=False)
                    
                    # Format for display
                    mech_df['Proportion'] = mech_df['Proportion'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(
                        mech_df,
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Sort for better visualization
                    sorted_mechs = dict(sorted(mech_profile.items(), key=lambda x: x[1], reverse=True))
                    
                    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_mechs)))
                    bars = ax.barh(list(sorted_mechs.keys()), list(sorted_mechs.values()), color=colors)
                    
                    ax.set_xlabel('Proportion', fontsize=10)
                    ax.set_title('Resistance Mechanism Distribution', fontsize=12, pad=20)
                    ax.set_xlim(0, 1)
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        if width > 0.05:  # Only label if significant
                            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.1%}', 
                                   va='center', fontsize=9)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Raw JSON in expander (for users who need it)
                with st.expander("📋 View Raw JSON Output"):
                    output = {
                        "Sample_ID": sample_id,
                        "AMR_Risk_Score": round(sample_score, 3),
                        "Risk_Category": risk_cat,
                        "Resistance_Mechanism_Profile": mech_profile,
                        "Interpretation": interpret(mech_profile)
                    }
                    st.code(json.dumps(output, indent=2), language="json")
                
                # Download button for this sample's results
                json_str = json.dumps(output, indent=2)
                st.download_button(
                    label=f"📥 Download Results for {sample_id}",
                    data=json_str,
                    file_name=f"amr_results_{sample_id}.json",
                    mime="application/json",
                    key=f"download_{sample_id}"
                )
                
                st.divider()
        
        # Batch download option
        st.markdown("### 📦 Batch Export")
        
        # Create comprehensive results
        all_results = []
        for i, sample_id in enumerate(df.index):
            mech_profile = compute_sample_mechanisms(df.loc[sample_id])
            all_results.append({
                "Sample_ID": sample_id,
                "AMR_Risk_Score": round(float(scores[i]), 3),
                "Risk_Category": risk_category(scores[i]),
                "Resistance_Mechanism_Profile": mech_profile,
                "Interpretation": interpret(mech_profile)
            })
        
        # Convert to DataFrame for CSV export
        results_df = pd.DataFrame(all_results)
        
        col1, col2 = st.columns(2)
        with col1:
            # Download as CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results (CSV)",
                data=csv,
                file_name="amr_results_all.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download as JSON
            json_all = json.dumps(all_results, indent=2)
            st.download_button(
                label="📥 Download All Results (JSON)",
                data=json_all,
                file_name="amr_results_all.json",
                mime="application/json"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please check your file format and try again.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Human Gut AMR Burden Prediction Tool • For Research Use Only</p>
    <p>Results should be interpreted by qualified healthcare professionals</p>
</div>
""", unsafe_allow_html=True)