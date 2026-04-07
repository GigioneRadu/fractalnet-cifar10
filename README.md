# 🧠 FractalNet: Deep Learning from Scratch

O implementare de la zero în **PyTorch** a arhitecturii FractalNet, antrenată pe dataset-ul **CIFAR-10**. Proiectul demonstrează cum putem antrena rețele neurale ultra-adânci fără a folosi conexiuni reziduale (ResNet), bazându-ne pe expansiunea fractală a blocurilor de convoluție.

## 🚀 Tehnologii folosite
* **Framework:** PyTorch
* **Cloud Training:** NVIDIA A4000 GPU (via Paperspace)
* **Interfață Vizuală:** Streamlit & Graphviz
* **Dataset:** CIFAR-10

## 📊 Arhitectură
Modelul folosește o regulă de expansiune recursivă pentru a crea căi paralele cu adâncimi diferite, permițând gradientului să circule eficient.


## 🛠️ Cum se rulează local
1. Instalează dependențele: `pip install -r requirements.txt`
2. Rulează aplicația: `streamlit run app.py`

## 📈 Rezultate
Modelul a atins o acuratețe de **~78%** în doar 15 epoci de antrenament "from scratch".