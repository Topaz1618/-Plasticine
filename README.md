<img src='images/logo.png' width='400' title='Plasticine, A facial deformation application'>

Plasticine is a facial deformation application based on face recognition, made by [Topaz](https://topaz1618.github.io/about)([Website](http://topazaws.com/)|[Blog](https://topaz1618.github.io/blog/))

[Chinese README](https://github.com/Topaz1618/Plasticine/blob/master/README_CN.md)


# Features:
- Support for obtaining key points of the face
- Support deformation of cheeks, chin, cheekbones


# Environment
- Python3
- tensorflow==1.15.0
- OpenCV
- Ubuntu16.04/macOS
- wxPython


## Installation (Ubuntu & macOS)
1. Download Plasticine
```
 git clone git@github.com:Topaz1618/Plasticine.git
```

2. Install dependencies
```
 pip install -r requirements.txt
```

## Run demo
```
 // The first frame needs to be initialized, so it may take a few seconds
 python slim_demo.py
```

## Run clinet
```
 python slim_client.py
```

If you have problem starting up on macOS, you can check this [wxPythonVirtualenvOnMac](https://wiki.wxpython.org/wxPythonVirtualenvOnMac)


## Face photo source
All test images are from this website, Click to view [AI Generate Face](https://generated.photos/faces/)

## Screenshots

## 【Face detection1】

Get the key points of the face (face deformation is based on the key points)
<img src='images/face.gif'  width='800' title='MeowShop, online shopping site'>

## 【Result1】
Push all parts of the face inward

<img src='static/images/slim.gif'  width='800' title='MeowShop, online shopping site'>

## 【Result2】

Pull all parts of the face outward

<img src='static/images/slim_face3.gif'  width='800' title='MeowShop, online shopping site'>


## 【Result3】

Cheeks become larger and the chin becomes smaller

<img src='static/images/smile.gif'  width='800' title='MeowShop, online shopping site'>



## License
Licensed under the MIT license