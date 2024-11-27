# Deep Neural Network Experiments with MNIST and Custom Datasets

This project explores the design, training, and evaluation of a Sequential neural network model for multi-class classification using the MNIST dataset and a custom dataset. The experiments focus on understanding the impacts of architecture and hyperparameter changes on model performance and generalization.

## **Overview**

The project involves:
- Designing a baseline Sequential model with Keras and TensorFlow.
- Experimenting with variations in the architecture and training parameters.
- Evaluating the model's performance on both MNIST and a custom dataset.
- Analyzing the trade-offs between accuracy, overfitting, and regularization.

## **Baseline Model**

The baseline model architecture consists of:
1. **Input Layer:** Flattens the input data.
2. **Hidden Layers:** Two dense layers with ReLU activation functions.
3. **Dropout Layer:** Applied for regularization.
4. **Output Layer:** Softmax activation for multi-class classification.

### **Performance**
- **10 Epochs Accuracy:** Achieved 99% accuracy on MNIST.
- **100 Epochs Accuracy:** Validation accuracy declined to 96%, indicating overfitting.

---

## **Experimental Configurations**

### **Model 1: Reduced Hidden Layer**
- **Change:** Removed one hidden layer from the baseline architecture.
- **Result:** Accuracy dropped to 10%, showing the need for sufficient model complexity.

### **Model 2: Increased Dropout**
- **Change:** Increased dropout rate to 0.5 to enhance regularization.
- **Result:** Stabilized at 93% accuracy, reducing overfitting but slightly impacting training performance.

### **Model 3: Batch Normalization**
- **Change:** Incorporated batch normalization layers for training stability.
- **Result:** Achieved 99% accuracy after 10 epochs but showed signs of overfitting.

---

## **Custom Dataset Testing**

The baseline model was evaluated on a custom dataset to assess its generalization capabilities. It achieved:
- **Accuracy:** 97%, aligning closely with the MNIST results.

---

## **Key Observations**

1. **Overfitting with Prolonged Training:** Training for 100 epochs caused accuracy to decline slightly, highlighting overfitting tendencies.
2. **Impact of Architecture and Regularization:**
   - Reducing model complexity led to poor performance.
   - Stronger regularization mitigated overfitting but reduced accuracy.
   - Batch normalization improved accuracy but increased overfitting risk.
3. **Generalization:** The model generalized well to unseen data, performing consistently across datasets.

---

## **Conclusion**

- Achieving optimal model performance requires balancing complexity and regularization.
- Architectural adjustments and hyperparameter tuning can substantially impact outcomes.
- Batch normalization is highly effective but needs careful monitoring to prevent overfitting.

---

## **Dependencies**

- Python >= 3.7
- TensorFlow >= 2.x
- Keras >= 2.x
- NumPy
- Matplotlib

Install dependencies via:
```bash
pip install tensorflow keras numpy matplotlib
