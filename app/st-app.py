""" Streamlit app for Radiacode gamma spectrum classification """


import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO
from spectrum import SpectrumPreprocessing
from radiacode import Spectrum
from models import SpectrumData
from ml_models import IsotopesClassificationModel
import logging
logger = logging.getLogger(__name__)


def get_spectrum_barchart_figure(sp: Spectrum) -> plt.Figure:
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


def is_xml_valid(xml_data: str) -> bool:
    """ Check if the XML has valid size and data """
    return len(xml_data) < 65535 and xml_data.startswith("<?xml")


def main():
    logger.info("App started")
    st.set_page_config(
        page_title='Gamma Spectrum', page_icon = Image.open("static/favicon.ico")
    )
    st.title("Radiacode Spectrum Detection (Beta)")
    st.text(
        "Export the spectrum to XML using the Radiacode app, and "
        "upload it to see the results."
    )
    st.text(
        "A model was trained on sources, available in homes (radium, thorium, "
        "uranium and americium). Logs will be analysed to improve the model."
    )

    description = st.text_input("Enter object description (optional):", "")

    uploaded_file = st.file_uploader("Choose the XML file", type="xml")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        xml_data = stringio.read()
        if is_xml_valid(xml_data):
            # Load
            sp = SpectrumPreprocessing.load_from_xml(xml_data)
            sp_str = SpectrumPreprocessing.to_string(sp)
            logger.info(f"Spectrum file loaded: '{description}';{sp_str}")

            # Prediction
            model = IsotopesClassificationModel()
            result = model.predict(SpectrumData(sp))
            logger.info(f"Spectrum prediction: {result}")

            # Show result
            st.success(f"Prediction Result: {result}")
            # Draw spectrum
            fig = get_spectrum_barchart_figure(sp)
            st.pyplot(fig)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
