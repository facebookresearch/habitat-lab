py-spy record --idle --function --native --subprocesses --rate 50 \
--output pyspy_profile.speedscope --format speedscope -- python \
habitat_baselines/run.py --exp-config \
habitat_baselines/config/pointnav/gala_kinematic.yaml --run-type \
train TOTAL_NUM_STEPS 200
