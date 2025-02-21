<a>
    <img src="https://readme-typing-svg.demolab.com?font=Georgia&size=50&duration=2000&pause=500&multiline=true&width=1500&height=80&lines=Geometric+Heuristics+Enhance+POCUS+AI+for+Pneumothorax" alt="Typing SVG" />
</a>



### Geometric Heuristics Enhance POCUS AI for Pneumothorax

**Viekash Vinoth Kumar, John Galeotti, Deva Kannan Ramanan, and Nishant Thumbavanam Arun**  
*Master's Thesis, Technical Report, CMU-RI-TR-24-11, May 2024*

[View Publication](https://www.ri.cmu.edu/publications/geometric-heuristics-enhance-pocus-ai-for-pneumothorax/)

Point-of-care ultrasound (POCUS) represents a major advancement in emergency and critical care medicine, delivering real-time imaging capabilities that are essential for rapid diagnostic and therapeutic decisions. Despite its transformative potential, interpreting POCUS images requires a high level of expertise, often leading to variability and potential diagnostic inaccuracies. Moreover, existing commercial POCUS AI systems typically demand large volumes of labeled training data, which can be challenging to acquire.

In our work, we addressed these challenges by training POCUS AI models with limited data while enhancing diagnostic accuracy through the direct integration of geometric heuristics into both the model architecture and training process. These heuristics, derived from expert clinical knowledge, encapsulate intuitive rules and patterns that clinicians rely on—such as observing the sliding of the pleural line, its relative movement against the intercostal muscles, and its precise anatomical positioning.

To effectively incorporate these insights, we represented the heuristics using:

- **Semantic Segmentation Label Images:**  
  These images highlight key anatomical regions by delineating areas such as the pleural line, ensuring that the model focuses on the most relevant features.

- **Optical Flow Images:**  
  Intended to capture the dynamic motion of the pleura during respiration, these images were used to provide additional temporal context. Although this approach did not yield the expected performance boost, it underscores the challenge of integrating motion-related data into POCUS interpretation.

Additionally, we cropped images based on the semantic segmentation maps to further focus the model’s attention on regions of interest (ROI). This targeted approach not only reduced noise in the training data but also emphasized clinically relevant features that are critical for diagnosing pneumothorax.

We developed and compared two distinct methods for embedding these heuristic images into our AI models:

- **Multi-Channel Input Approach:**  
  In this method, the original grayscale ultrasound images were augmented with additional channels containing heuristic information. This allowed the model to process spatial details and motion-related cues simultaneously from the very beginning of the training process.

- **Fused Embedding Space Approach:**  
  Here, heuristic images were incorporated as separate inputs that were later merged into a common embedding space alongside the original image data. This modular integration aimed to synthesize complementary information from both streams, ultimately forming a unified representation for diagnosis.

Our experimental results demonstrated that the strategies involving ROI cropping and the use of semantic segmentation maps significantly outperformed baseline models. Notably, while the integration of optical flow maps did not enhance performance as initially hypothesized—highlighting the nuanced challenge of effectively incorporating motion cues—the multi-channel input approach offered a slight performance advantage over the fused embedding space method.

Overall, this study demonstrates that integrating expert-derived geometric heuristics can substantially improve the diagnostic accuracy of AI systems interpreting POCUS images for pneumothorax. The findings highlight the value of embedding domain-specific knowledge into the AI model, especially when working with limited training data. Moreover, this research paves the way for exploring additional heuristic combinations and refining model architectures to further enhance performance, particularly in challenging applications such as POCUS video interpretation, where both clinicians and AI systems often struggle to discern subtle anatomical and motion cues.
