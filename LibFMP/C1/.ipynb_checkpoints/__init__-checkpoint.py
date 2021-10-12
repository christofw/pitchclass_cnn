from .C1S1_SheetMusic import generate_sinusoid_pitches, \
    generate_shepard_tone, \
    generate_chirp_exp_octave, \
    generate_Shepard_glissando

from .C1S2_SymbolicRep import csv_to_list, \
    midi_to_list, \
    xml_to_list, \
    list_to_csv, \
    visualize_piano_roll

from .C1S3_AudioRep import F_pitch, \
    difference_cents, \
    generate_sinusoid, \
    compute_power_dB, \
    compute_equal_loudness_contour, \
    generate_chirp_exp, \
    generate_chirp_exp_equal_loudness, \
    compute_ADSR, compute_envelope, \
    compute_plot_envelope, \
    generate_sinusoid_vibrato, \
    generate_sinusoid_tremolo, \
    generate_tone
