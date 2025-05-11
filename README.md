# florr auto afk (v1.2.5) (2025-05-11 SSL export Update)

> As m28 released his new anti afk, I‘ll put my new anti-anti afk code here.

> [!CAUTION]
> 
>M28 can send player messages and AFK Checks, if you can solve the check but cannot respond to his messages, you may result in a BAN

## INFO

For the latest v1.2.x , you need **WINDOWS** device to run, or you can only use v1.1.1.

After trying the onnx models, I think it's the worst decision I have ever made.

Now those stupid codes go under the `onnx` branch

Latest v1.2.x version, I added background AFK Check detection support, this requires your browser supports disabling

`CalculateNativeWinOcclusion` or try to use [Firefox](https://www.mozilla.org/en-US/firefox) browser

### Chrome

Go to `chrome://flags/` and search for `CalculateNativeWinOcclusion` if this appears go `disable` it

If you can't find the element, try run this and replace the `chrome.exe` path if needed

```shell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --disable-features=CalculateNativeWinOcclusion
```

### Edge

I didn't find a method to disable the future, however, I found this in `Edge Beta` , `Edge Dev` and `Edge Canary`, check the Insider Edge version at https://www.microsoft.com/en-us/edge/download/insider

After you install the Insider version, go to `edge://flags/`  and do the same thing as **Chrome** does

### Firefox

Congratulations, Firefox disable  `CalculateNativeWinOcclusion` by default, enjoy your game in Firefox

## Note

I have opensourced the code here.

If you want to write more automation codes, go check models in [assets](https://github.com/Shiny-Ladybug/assets)

If you want to boot this without an `Internet Connection`, try to set `skipUpdate` in `config.json` to `true`.

## Changelog

* 2025-05-04

  If you turn on SaveTrainableDataset in Settings > Advanced, program will save the dataset to `./train` folder, I'm welcomed to receive those datasets to improve the AFK model

* 2025-04-26

  Add background AFK Check detection support
* 2025-04-18

  Add GUI, exposure, idle detection support

  As that exposure can nerf the mobs' movement effect and idle detection enables me to active (maybe kill supers) without quitting the program

  ![exposure showcase](./imgs/exposure.jpg)

  ![GUI Settings](./imgs/settings.png)

## Deploy Locally

```bash
pip install -r ./py311-requirements.txt
python segment.py
```

Notice: the release version is only for **WINDOWS** and **CPU ONLY** users.

If you want to run the script on MacOS or Linux, go to run source codes.

You can build your own GPU version here by installing `torch-gpu`.

## Config	

See the settings page in GUI menu.

If you can't understand what's this and you are Chinese by accident, set `language` to `zh-cn` for Chinese settings.

## Gallery

![img](./imgs/gallery.png)

## How it works

1. Use PaddleOCR to detect if the screen contains "AFK Check" (deprecated as m28 could send a window contain no "AFK Check" text)

   OCR model using `ch_pp_ocrv3`

   I know that I can use a tampermonkey script which rewrites the `canvas.FillText`

   ```js
   function rewriteFillText() {
           function getCompatibleCanvas() {
               if (typeof (OffscreenCanvasRenderingContext2D) == 'undefined') {
                   return [CanvasRenderingContext2D]
               }
               return [OffscreenCanvasRenderingContext2D, CanvasRenderingContext2D];
           }
           const idSymbol = Symbol('id');
           for (const {prototype} of getCompatibleCanvas()) {
               prototype[idSymbol] = prototype.fillText
           }
           for (const {prototype} of getCompatibleCanvas()) {
               prototype.fillText = function (text, x, y) {
                   // DO SOMETHING NASTY
                   return this[idSymbol](text, x, y);
               }
               prototype.fillText.toString = () => 'function toString() { [native code] }';
           }
   
       }
   ```

   I can start a local HTTP API to see if i got checked.

   For specified reasons, I do not recommend using internal scripts (for unknown BAN results). I'd like doing all these tasks by Python.

   So I used a YOLO model `afk-det.pt` to detect the AFK Check windows.
2. Trying using yolo model `afk-seg`.pt to separate the mouse path.

   ![results.png](./imgs/results.png)

   Obviously i got a good model for this.

   After I get the contours, we can use the `cv2.ximgproc.thinning()` method to get the skeletonized path.
3. Sometimes the yolo model cannot detect the possible results.

   I use opencv as well to detect the path.

   By using the Grey Style, we can define a specific `lower_bounds` and `upper_bounds` to get the path.
4. Loop

## FAQ

Q: What is the accuracy of this model?

A: According to statistics, after properly installing the newest version of Auto AFK, the accuracy should be greater than **90%**.

Q: Why the accuracy tested on my computer is so low?

A: Most likely skill issue. Make sure you are NOT using normal Edge (mentioned above). Also make sure you have disabled `CalculateNativeWinOcclusion` if you're on Google Chrome or other versions of Edge.

Q: Why is my computer so hot when running the code?

A: Make sure you close all the unnecessary programs when running and maybe turn on "Best Battery Life".

Q: But my computer goes to sleep after a very short period of time. How to fix that?

A: Go to Settings in your computer and change the sleep after inactive time of course.

Q: I found some bugs/I have some ideas for improving this application. What should I do?

A: Suggest turning on `Save Trainable Dataset` in Settings > Advanced. Then contact Shiny Ladybug via QQ.

## Issues

**I DO NOT RECOMMEND TRUSTING THIS SCRIPT**

It can passing some AFK checks.

As the longer you stay in the same server, the checks get harder.

The script cannot solve the ***WORM-LIKE*** disgusting checks for the time being.

## ?

<img src="./imgs/39ca67e4e7f587a7d8f7c3284c344d0e.png" width="600" />

Some people queried why I was using GPLv3 without opensourcing.

BRO,that night was **TRADITIONAL CHINESE NEW YEAR's EVE**, can't I just upload the release code and have a break at this good time. Or you can read the release notes carefully words by words and you can find the problem.
