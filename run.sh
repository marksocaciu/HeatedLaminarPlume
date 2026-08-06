
python postprocess_steady_plume.py \
 --temperature-xdmf PlumeCase_Brodowicz_Air_reduced_2/runs/abe/base/air_temperature_steady_from_transient_step_14360.xdmf \
  --velocity-xdmf PlumeCase_Brodowicz_Air_reduced_2/runs/abe/base/air_velocity_steady_from_transient_step_14360.xdmf \
  --heatflux-xdmf PlumeCase_Brodowicz_Air_reduced_2/runs/abe/base/air_temperature_heatflux_final_steady_14360.xdmf \
  --pressure-xdmf PlumeCase_Brodowicz_Air_reduced_2/runs/abe/base/air_pressure_steady_from_transient_step_14360.xdmf \
  --outdir PlumeCase_Brodowicz_Air_reduced_2/runs/abe/base/postprocess_thesis_2 \
  --theory-profile-csv PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/postprocess_thesis/fujii_line_plume_theory_current_case.csv \
  --coords-are-dimensionless \
  --lref 3.75e-05 \
  --T-inf 292.95 --k 0.0257 \
  --rho 1.1614 \
  --cp 1007.0 \
  --mu 1.85e-5 \
  --beta 0.0034 \
  --q-input-per-length 9.75 \
  --planes 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08\
  --profile-half-width 0.05 \
  --eta-plot-half-width 5 \
  --energy-eta-half-width 5 \
  --plot-width-inch 8 \
  --plot-font-size 18 \
  --energy-cv \
  --energy-cv-width-mode eta \
  --energy-cv-eta-half-width 5 \
  --energy-cv-min-half-width-m 1.5913e-2 \
  --nusselt \
  --solid-k 10.0  --plot-width-inch 16.0 --plot-height-inch 6.0


python postprocess_steady_plume.py \
  --temperature-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/air_temperature_steady_from_transient_step_02700.xdmf \
  --velocity-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/air_velocity_steady_from_transient_step_02700.xdmf \
  --heatflux-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/air_temperature_heatflux_final_steady_02700.xdmf \
  --pressure-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/air_pressure_steady_from_transient_step_02700.xdmf \
  --outdir PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/abe/base/postprocess_thesis_htff26_fujii \
  --theory-profile-csv PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/postprocess_thesis/fujii_line_plume_theory_current_case.csv \
  --coords-are-dimensionless \
  --lref 7.14285714e-04 \
  --T-inf 292.95 \
  --rho 1.1614 \
  --cp 1007.0 \
  --k 0.0257 \
  --mu 1.85e-5 \
  --beta 0.0034 \
  --q-input-per-length 0.2687 \
  --planes 0.005 0.01 0.02 0.03 0.035\
  --profile-half-width 0.05 \
  --eta-plot-half-width 5 \
  --energy-eta-half-width 5 \
  --plot-width-inch 8 \
  --plot-font-size 18 \
  --energy-cv \
  --energy-cv-width-mode eta \
  --energy-cv-eta-half-width 5 \
  --energy-cv-min-half-width-m 1.5913e-2 \
  --nusselt \
  --solid-k 10.0  --plot-width-inch 16.0 --plot-height-inch 6.0



python postprocess_steady_plume.py \
  --temperature-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/air_temperature_steady_from_transient_step_06600.xdmf \
  --velocity-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/air_velocity_steady_from_transient_step_06600.xdmf \
  --heatflux-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/air_temperature_heatflux_final_steady_06600.xdmf \
  --pressure-xdmf PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/air_pressure_steady_from_transient_step_06600.xdmf \
  --outdir PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/postprocess_thesis \
  --theory-profile-csv PlumeCase_DeschampsDesrayaud_Air_Cylinder_Ra1e6/runs/base/base/postprocess_thesis/fujii_line_plume_theory_current_case.csv \
  --coords-are-dimensionless \
  --lref 7.14285714e-04 \
  --T-inf 292.95 \
  --rho 1.1614 \
  --cp 1007.0 \
  --k 0.0257 \
  --mu 1.85e-5 \
  --beta 0.0034 \
  --q-input-per-length 0.2687 \
  --planes 0.005 0.01 0.02 0.03 0.035 0.04 0.05 0.06 0.07 0.08 \
  --profile-half-width 0.05 \
  --eta-plot-half-width 5 \
  --energy-eta-half-width 5 \
  --plot-width-inch 8 \
  --plot-font-size 18 \
  --energy-cv \
  --energy-cv-width-mode eta \
  --energy-cv-eta-half-width 5 \
  --energy-cv-min-half-width-m 1.5913e-2 \
  --nusselt \
  --solid-k 10.0  --plot-width-inch 16.0 --plot-height-inch 6.0
