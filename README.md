<a name="readme-top"></a>

<h1 align="center">
  <br>
  <img src="https://raw.githubusercontent.com/MichlF/sports_object_detection/tree/main/readme_images/tennis.png" title="From Flaticon Those Icons" alt="From Flaticon Those Icons" width="75"></a>
  <br>
  Object Detection in Sports
  <br>
</h1>

<h4 align="center">A Tennis tracker project.</h4>

<h1 align="center">

  [![Contributors][contributors-shield]][contributors-url]
  [![Forks][forks-shield]][forks-url]
  [![Stargazers][stars-shield]][stars-url]
  [![Issues][issues-shield]][issues-url]

</h1>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

<p align="center">
    <a href="https://github.com/MichlF/sports_object_detection/issues">Report Bug</a> •
    <a href="https://github.com/MichlF/sports_object_detection/issues">Request Feature</a>
</p>

![screenshot](https://raw.githubusercontent.com/MichlF/sports_object_detection/tree/main/readme_images/demo.gif)

## Key Features

* Detects the players and the tennis ball in a given video
* Tracks player and ball positions and stores them for later analysis
* Ticker-style recreation of the match onto a top-view minimap

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How To Use

For this app to work you need to either clone or [Git](https://git-scm.com) install this repo on your computer. Make sure to install the requirements along with it. You can do this by running the following command in your command line:

```bash
# Pip install with requirements
pip install git+https://github.com/MichlF/sports_object_detection.git -r requirements.txt
```

In your favorite IDE or in a simple text editor, you can now change the parameters in the `config.py` file. The most critical is the `VIDEO_IN_NAME`. Change this to your tennis match mp4 file and store that video in the **input** folder. Then run the `main.py` file either from your IDE or the terminal:

```bash
python main.py
```

> **Note:**
> Running this requires **Tensorflow** and **PyTorch**. You may need to install them separately. You can learn how to install Tensorflow [here](https://www.tensorflow.org/install) and PyTorch [here](https://pytorch.org/get-started/locally/) or run this on a cloud service such as [Google Colab](https://colab.research.google.com/) or [DeepNote](https://deepnote.com/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## How it works

EMPTY

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap  

The following features are in the making ([contributions welcome !](#contributing)) and limitations apply.
### Features to add  

- [ ] Provide an extensive explanation on how it works
- [ ] Implement saving the ball trajectory and player path to .csv
- [ ] Provide additional metrics
  - [ ] Provide a counter for the number of ball bounces
  - [ ] Provide a counter for the number of strokes
  - [ ] Calculate speed for each ball stroke
- [ ] Allow passing arguments overriding config arguments when running from CLI
- [ ] Instead of using TrackNet, create a custom dataset and train YOLO to detect the tennis ball reliably
- [ ] Add transformations for video footage with a moving camera
- [ ] Improve player vs. non-player detection 
- [ ] Provide code adaptations to fully support other sports

### Limitations  

Because we only have a single camera, certain information, such as depth or height of an object, cannot be reliably inferred. This makes the match recreation on the minimap and certain (physical) metrics inaccurate. Similarly, everything hinges on accuracy of court line and general object detection which means that the quality of your video footage is a major determinant of the output quality ("garbage in, garbage out").

### More features? Problems? Bugs?  
See the [open issues](https://github.com/MichlF/sports_object_detection/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing  

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgements  
- YOLOv8 from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Huang, Y. C., Liao, I. N., Chen, C. H., İk, T. U., & Peng, W. C. (2019, September). TrackNet: A deep learning network for tracking high-speed and tiny objects in sports applications. In 2019 16th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS) (pp. 1-8). IEEE. Find the repository [here](https://nol.cs.nctu.edu.tw:234/open-source/TrackNet)  
- Maxime Bataille's version (find it [here](https://github.com/MaximeBataille/tennis_tracking))
- levan92's [DeepSort](https://github.com/levan92/deep_sort_realtime) in realtime  


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

> [blog.michelfailing.de](https://blog.michelfailing.de) &nbsp;&middot;&nbsp;
> GitHub [@MichlF](https://github.com/MichlF) &nbsp;&middot;&nbsp;
> Twitter [@FailingMichel](https://twitter.com/FailingMichel)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/MichlF/sports_object_detection.svg?style=flat
[contributors-url]: https://github.com/MichlF/sports_object_detection/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MichlF/sports_object_detection.svg?style=flat
[forks-url]: https://github.com/MichlF/sports_object_detection/network/members
[stars-shield]: https://img.shields.io/github/stars/MichlF/sports_object_detection.svg?style=flat
[stars-url]: https://github.com/MichlF/sports_object_detection/stargazers
[issues-shield]: https://img.shields.io/github/issues/MichlF/sports_object_detection.svg?style=flat
[issues-url]: https://github.com/MichlF/sports_object_detection/issues
[license-shield]: https://img.shields.io/github/license/MichlF/sports_object_detection.svg?style=flat
[license-url]: https://github.com/MichlF/sports_object_detection/blob/master/LICENSE.txt