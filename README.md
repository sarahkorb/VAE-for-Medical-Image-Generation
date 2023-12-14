# VAE-for-Medical-Image-Generation

This is a repository for a research project by Duke students Teo Feliu, Maximilian Holsman, and Emmanuel Mokel. The project aimed to find ways in which we can use Variational Autoencoders (VAEs) to generate high quality medical images. Our full project report can be found [here](https://github.com/maxholsman/VAE-for-Medical-Image-Generation/blob/main/CS_675_Final_Project_Writeup.pdf). 

This project used code from the [PyTorch-VAE repository](https://github.com/AntixK/PyTorch-VAE) and the [VAE-GAN-PYTORCH repository](https://github.com/rishabhd786/VAE-GAN-PYTORCH). 

# Example Generations
<div style="display: flex; flex-wrap: wrap;">
    <div style="width: 50%;">
        <h2>Training Images</h2>
        <img src="./examples.png" alt="Training Examples" style="width: 100%;">
    </div>
    <div style="width: 50%;">
        <h2>Generated Images</h2>
        <img src="./Prostate_VAEGAN.png" alt="Generated Images" style="width: 100%;">
    </div>
</div>

# Abstract
Variational Autoencoders (VAEs) (Kingma & Welling, 2022) are a deep generative model used to produce realistic synthetic data in the same theme as training data. Collecting data in a medical context can be a time consuming and expensive task, making it difficult to make predictive and interpretable machine learning models in medical contexts. VAEs can be used to improve the quality of medical image datasets, which may be of low resolution or contain lots of noise. We utilize several VAE variants in order to generate images from several medical datasets. We then train a generator to distinguish synthetic training data created by our VAEs from real medical data, similar to how Generative Adversarial Networks (GANs) work, in order to improve our images. We hope this method can be applied to generate synthetic medical images for real world use, and improve
existing data.


