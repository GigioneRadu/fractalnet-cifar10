import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import graphviz
import os

# Importăm arhitectura din modulul creat 
from src.architecture import FullFractalNet

st.set_page_config(page_title="FractalNet Portfolio", layout="wide")

st.title("FractalNet: Ultra-Deep Networks without Residuals 🧠")
st.markdown("""
**Proiect End-to-End creat pentru LinkedIn.** Această aplicație demonstrează arhitectura FractalNet. Spre deosebire de ResNet (care folosește skip connections), 
FractalNet previne dispariția gradientului printr-o regulă de expansiune recursivă, creând o arhitectură auto-similară.
""")

st.markdown("---")

# ==========================================
# SECTIUNEA 1: VIZUALIZATOR DE ARHITECTURĂ
# ==========================================
st.header("1. Explorează Topologia Fractală")

def build_fractal_graph(depth, graph, parent_node, prefix):
    """Generează graful recursiv pentru vizualizare"""
    if depth == 1:
        node_name = f"{prefix}_conv"
        graph.node(node_name, "Conv\nBlock", style="filled", fillcolor="#ADD8E6", shape="box")
        graph.edge(parent_node, node_name)
        return node_name
    else:
        left_out = build_fractal_graph(depth - 1, graph, parent_node, f"{prefix}_L")
        
        right_mid = build_fractal_graph(depth - 1, graph, parent_node, f"{prefix}_R1")
        right_out = build_fractal_graph(depth - 1, graph, right_mid, f"{prefix}_R2")
        
        join_node = f"{prefix}_join"
        graph.node(join_node, "Mean", style="filled", fillcolor="#FFD700", shape="circle")
        
        graph.edge(left_out, join_node)
        graph.edge(right_out, join_node)
        
        return join_node

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Ajustează complexitatea")
    depth = st.slider("Adâncimea fractală ($c$):", 1, 4, 2)
    st.latex(r"f_{c+1}(x) = \frac{1}{2} \Big[ f_c(x) + f_c(f_c(x)) \Big]")
    st.info("Mută slider-ul pentru a vedea cum rețeaua se extinde recursiv, ramificându-se în sub-rețele de adâncimi diferite.")

with col2:
    dot = graphviz.Digraph(comment='FractalNet')
    dot.attr(rankdir='TB', size='8,8')
    dot.node('input', 'Input', shape='ellipse')
    last_node = build_fractal_graph(depth, dot, 'input', 'root')
    dot.node('output', 'Output', shape='ellipse')
    dot.edge(last_node, 'output')
    st.graphviz_chart(dot)

st.markdown("---")

# ==========================================
# SECTIUNEA 2: TESTEAZĂ MODELUL ANTRENAT
# ==========================================
st.header("2. Testează Modelul (Antrenat pe Paperspace A4000)")

st.markdown("Am antrenat un model FractalNet (Adâncime 3) de la zero pe datasetul CIFAR-10 folosind un GPU NVIDIA A4000. Încarcă o imagine pentru a vedea predicția!")

uploaded_file = st.file_uploader("Încarcă o imagine (ex: mașină, câine, avion, pisică, vapor)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col_img, col_pred = st.columns(2)
    
    with col_img:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Imaginea ta', use_column_width=True)
    
    with col_pred:
        st.markdown("### Rezultat AI")
        # Transformările identice cu cele folosite la evaluare
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        input_tensor = transform(image).unsqueeze(0)
        
        model_path = 'models/fractalnet_cifar10.pth'
        
        if os.path.exists(model_path):
            with st.spinner('Analizez imaginea...'):
                # Încărcăm modelul (pe CPU, deoarece rulăm local interfața)
                model = FullFractalNet(num_classes=10)
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                
                classes = ['Avion', 'Automobil', 'Pasăre', 'Pisică', 'Cerb', 'Câine', 'Broască', 'Cal', 'Vapor', 'Camion']
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    top_prob, top_class = torch.max(probabilities, 0)
                    
                st.success(f"**Predicție:** {classes[top_class.item()]}")
                st.info(f"**Încredere:** {top_prob.item() * 100:.2f}%")
        else:
            st.error(f"Nu găsesc fișierul greutăților la calea: `{model_path}`. Asigură-te că l-ai descărcat de pe Paperspace!")
