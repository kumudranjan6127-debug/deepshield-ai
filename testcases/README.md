# Images that were called wrong

Real failures beat synthetic ones. Everything in this folder is a picture the
deployed app got wrong, kept so a fix can be checked against the thing that
actually broke rather than a composite built to resemble it.

Drop a file in and run:

    venv/Scripts/python scripts/check_case.py testcases/<file>

Not committed - these are usually someone's photograph. `.gitignore` covers
the folder and keeps this README.
