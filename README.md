<img src='static/images/meowshop.png' width='400' title='Plasticine, A facial deformation application'>

Plasticine is a facial deformation application based on face recognition, made by [Topaz](https://topaz1618.github.io/about)([Website](http://topazaws.com/)|[Blog](https://topaz1618.github.io/blog/))

[Chinese README](https://github.com/Topaz1618/MeowShop/blob/master/README_CN.md)


# Features:
- Support real-time adjustment
- Support deformation of cheeks, chin, cheekbones


# Environment
- Python3
- wxPython
- tensorflow==1.15.0
- OpenCV
- Ubuntu16.04/macOS


## Installation (Ubuntu & Mac OS)
1. Download MeowFile
```
 git clone git@github.com:Topaz1618/MeowShop.git
```

2. Install dependencies
```
 pip install -r requirements.txt
```

## Run demo
```
 // Be patient, the first frame needs to be initialized, the test case only processes one frame, so it may take a few seconds, You can also optimize, use your GPU, or add multi-threading
 python slim_demo.py
```

## Run clinet
```
 python slim_client.py
```

If you have problem starting up on macOS, you can check this [wxPythonVirtualenvOnMac](https://wiki.wxpython.org/wxPythonVirtualenvOnMac)


# Face photo source
Click to view [AI Generate Face](https://generated.photos/faces/)

## Screenshots

## 【Result1】
<img src='static/images/store222.gif'  width='800' title='MeowShop, online shopping site'>

## 【Result2】
<img src='static/images/store22.gif'  width='800' title='MeowShop, online shopping site'>

## 【Result3】
<img src='static/images/pay.gif'  width='800' title='MeowShop, online shopping site'>



## License
Licensed under the MIT license