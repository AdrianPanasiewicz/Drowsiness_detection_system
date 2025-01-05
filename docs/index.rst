.. System wykrywania senności documentation master file, created by
   sphinx-quickstart on Sun Jan  5 15:38:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dokumentacja systemu wykrywania senności
========================================

Autor: Adrian Paweł Panasiewicz

Tytuł pracy dyplomowej:
Projekt wstępny systemu bezpieczeństwa do wykrywania senności u pilotów bezzałogowych statków powietrznych

Cel aplikacji
-------------

Aplikacja została zaprojektowana w celu monitorowania stanu senności
operatorów dronów w czasie rzeczywistym. Analizuje obraz z kamery
i wyświetla parametry takie jak PERCLOS, ziewanie czy pochylenie
głowy. W przypadku wykrycia krytycznych wartości system generuje
alerty dźwiękowe i wizualne.

Jak obsługiwać aplikację
------------------------

1. Uruchom plik wykonywalny aplikacji na systemie Windows 10/11.

2. Podłącz kamerę zgodną z minimalnymi wymaganiami (720p, 30 FPS).

3. Ustaw kamerę tak, aby rejestrowała twarz operatora w dobrych warunkach oświetleniowych.

4. Obserwuj dane wyjściowe na interfejsie graficznym aplikacji (GUI).

**Uwaga:**

1. Aplikacja obsługuje tylko jedną twarz w kadrze.
2. Stabilne oświetlenie i minimalne ruchy kamery poprawiają dokładność analizy.
3. Wszelkie dane są zapisywane w bazie danych, umożliwiając późniejszą analizę.

Instalacja
----------

1. Pobierz repozytorium z GitHub

.. code-block:: bash

    git clone https://github.com/AdrianPanasiewicz/Drowsiness_detection_system.git

2. Zainstaluj potrzebne paczki

.. code-block:: bash

    pip install -r requirements.txt

.. toctree::
   :maxdepth: 5
   :caption: Zawartość:

   modules

Indeksy i tabele
================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
