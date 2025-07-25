""" Streamlit app for Radiacode gamma spectrum classification """


import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO
import os
from typing import Optional
from spectrum import SpectrumPreprocessing
from models import SpectrumData, Spectrum
from ml_models import IsotopesClassificationModel
import logging
logger = logging.getLogger(__name__)


def get_page_icon() -> Image:
    """ Get app icon """
    path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Local path: {path}")
    return Image.open(path + "/static/favicon.ico")


def is_xml_valid(xml_data: str) -> bool:
    """ Check if the XML has valid size and data """
    return len(xml_data) < 65535 and xml_data.startswith("<?xml")


def get_spectrum(stringio: StringIO) -> Optional[Spectrum]:
    """ Load spectrum from the StringIO stream """
    xml_data = stringio.read()
    if is_xml_valid(xml_data):
        return SpectrumPreprocessing.load_from_xml(xml_data)
    return None


def get_spectrum_barchart(sp: Spectrum) -> plt.Figure:
    """ Get Matplotlib's barchart """
    counts = SpectrumPreprocessing.get_counts(sp)
    energy = [SpectrumPreprocessing.channel_to_energy(sp, x) for x in range(len(counts))]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.spines["top"].set_color("lightgray")
    ax.spines["right"].set_color("lightgray")
    # Bars
    ax.bar(energy, counts, width=3.0, label="Counts")
    # X values
    ticks_x = [SpectrumPreprocessing.channel_to_energy(sp, ch) for ch in range(0, len(counts), len(counts) // 20)]
    labels_x = [f"{int(ch)}" for ch in ticks_x]
    ax.set_xticks(ticks_x, labels=labels_x, rotation=45)
    ax.set_xlim(energy[0], energy[-1])
    ax.set_ylim(0, None)
    ax.set_title("Gamma spectrum")
    ax.set_xlabel("Energy, keV")
    ax.set_ylabel("Counts")
    return fig


def main():
    logger.info("App started")
    st.set_page_config(
        page_title='Gamma Spectrum', page_icon=get_page_icon()
    )
    st.title("Radiacode Spectrum Detection (Beta)")
    st.text(
        "Export the spectrum to XML using the Radiacode app, and "
        "upload it to see the results."
    )
    st.text(
        "V1: A model was trained on sources, available in homes (radium, thorium, "
        "uranium and americium).\nFeel free to test other sources (please add description), "
        "logs will be analysed to improve the model."
    )
    st.text_input(
        "Enter object description (optional):", "", key="object_description"
    )

    def on_file_uploader_changed():
        """ If the file was already uploaded, clear the object description """
        # Save spectrum to log
        uploader = st.session_state['uploader']
        stringio = StringIO(uploader.getvalue().decode("utf-8"))
        if sp := get_spectrum(stringio):
            # Add description
            description = st.session_state["object_description"]
            sp_str = SpectrumPreprocessing.to_string(sp)
            logger.info(f"Spectrum file loaded: '{description}';{sp_str}")

        # Clear description for a next use
        st.session_state["object_description"] = ""


    uploaded_file = st.file_uploader(
        "Choose the XML file", type="xml", key="uploader", on_change=on_file_uploader_changed
    )
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # Load from uploaded file
        if sp := get_spectrum(stringio):
            # Prediction
            model = IsotopesClassificationModel()
            result = model.predict(SpectrumData(sp))
            logger.info(f"Spectrum prediction: {result}")

            # Show result
            st.success(f"Prediction Result: {result}")
            # Draw
            fig = get_spectrum_barchart(sp)
            st.pyplot(fig)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
