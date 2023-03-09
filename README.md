# HistDeblur: A Pyramid Trustworthy Framework for Blurry Histologic Artifacts Quality Control

To train a new model, run the following command:

```bash
python train.py
```

The checkpoint will be saved at ``.\checkpoints`` 

To evaluate the model's performance or deblur, set the correct checkpoint root and mode in ``eval.py`` and run the following command:

```bash
python eval.py
```

More detailed description about the options will be found in  ``arguments/arguments_train.py`` and ``eval.py``
