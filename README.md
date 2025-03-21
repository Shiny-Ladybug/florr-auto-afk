# florr auto afk (v1.0.7) (2025-03-16 Update)

> As m28 released his new anti afk, I‘ll put my new anti-anti afk code here.


> [!CAUTION]
> On 2025/02/07, m28 nerfed egg and increase the difficulty of AFK Checks, however, keep using this script may result in a BAN

## INFO

I won't put the latest obfuscated code here for some reason. Use this script in RELEASE.

If you want to boot this without an `Internet Connection`, try to set `skipUpdate` in `config.json` to `true`.

## Config

At `config.json`

* `runningCountDown(min)` 

  Set it to `-1` to let the script run endlessly untill you press `^C` 

  Or set it to a Positive Number to run for a definite time

* `useOBS`

  Just for fun. Set OBS Start hotkey to `Ctrl Alt Shift O` and End hotkey to `Ctrl Alt Shift P` to capture the amazing moment of passing AFK Checks.

* `moveAfterAFK`

  Make the player move a bit `wasd` after checks to prevent `Moving Checks`

* `epochInterval`

  How many seconds should the program sleep after one check. Decrease this value if you have anxiety disorders

* `extraBinary` 

  If you have something to excute before or after AFK Checks detected, you can write the executables path here.

I DO NOT RECOMMEND CHANGING THE REST OF THE VALUES, CHANGING THOSE MAY CAUSE ISSUE IN PROGRAM

I WONT EXPLAIN THEM NOW, FOR I HAVE TO SLEEP RIGHT AWAY.

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

​		Obviously i got a good model for this.

​		After I get the contours, we can use the `cv2.ximgproc.thinning()` method to get the skeletonized path.

3. Sometimes the yolo model cannot detect the possible results.

   I use opencv as well to detect the path.

   By using the Grey Style, we can define a specific `lower_bounds` and `upper_bounds` to get the path.

4. Loop

## Deploy

```bash
pip install -r ./py311-requirements.txt
python segment.py
```

Anyway, the release version is only for **WINDOWS** and **CPU ONLY** users.

You can build your own GPU version here by installing `torch-gpu`.

## Issues

**I DO NOT RECOMMEND TRUSTING THIS SCRIPT**

It can really passing some easy AFK checks.

As the longer you stay in the same server, the checks get harder.

The script cannot solve the ***WORM-LIKE*** disgusting checks for the time being.

I'll improve the code soon.

## ?

<img src="./imgs/39ca67e4e7f587a7d8f7c3284c344d0e.png" width="600" />

Some people queried why I was using GPLv3 without opensourcing.

BRO,that night was **TRADITIONAL CHINESE NEW YEAR's EVE**, can't I just upload the release code and have a break at this good time. Or you can read the release notes carefully words by words and you can find the problem.
