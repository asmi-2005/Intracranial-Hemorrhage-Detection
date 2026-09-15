import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------
# CHANGE IMAGE PATH
# -------------------------------------------------------

image_path = r"C:\Users\ASMI\OneDrive\Desktop\Intracranial_Hemorrhage_Detection\dataset\Patients_CT\057\brain\22.jpg"

# -------------------------------------------------------
# Dummy Prediction
# -------------------------------------------------------

prediction = "Intraparenchymal Hemorrhage"
confidence = 94.82
risk = "High"

recommendation = (
    "Possible intracranial hemorrhage detected.\n"
    "Immediate radiological review is recommended."
)

# -------------------------------------------------------
# Display
# -------------------------------------------------------

image = Image.open(image_path)

plt.figure(figsize=(10,8))
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.title(
    f"""
AI-Based Intracranial Hemorrhage Detection

Predicted Class : {prediction}
Confidence      : {confidence:.2f}%
Risk Level      : {risk}

Recommendation:
{recommendation}
""",
fontsize=12,
fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/demo_prediction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("="*60)
print(" Intracranial Hemorrhage Detection Result")
print("="*60)
print(f"Predicted Class : {prediction}")
print(f"Confidence      : {confidence:.2f}%")
print(f"Risk Level      : {risk}")
print("Recommendation  : Immediate radiological review advised.")
print("="*60)