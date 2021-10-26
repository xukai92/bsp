Threads.@threads for fn_id in 1:60
    cmd = `python scripts/preprocess-ullman.py $fn_id`
    println("Running: $cmd")
    run(cmd)
end

# o: broken before but corrected now
# ?: resolvable by tuning parameters
# x: ambiguous scenes
# [
#     'world1_1.mp4', # 1
#     'world1_2.mp4',
#     'world1_3.mp4',
#     'world1_4.mp4',
#     'world1_5.mp4',
#     'world1_6.mp4',

#     'world2_1.mp4', # 7
#     'world2_2.mp4', o overlap 8
#     'world2_3.mp4',
#     'world2_4.mp4',
#     'world2_5.mp4',
#     'world2_6.mp4',

#     'world3_1.mp4', # 13
#     'world3_2.mp4',
#     'world3_3.mp4',
#     'world3_4.mp4', o overlap 16
#     'world3_5.mp4',
#     'world3_6.mp4',

#     'world4_1.mp4', # 19
#     'world4_2.mp4',
#     'world4_3.mp4', ? overlap 21
#     'world4_4.mp4', x overlap 22
#     'world4_5.mp4',
#     'world4_6.mp4',

#     'world5_1.mp4', o overlap 25
#     'world5_2.mp4',
#     'world5_3.mp4', o overlap 27
#     'world5_4.mp4',
#     'world5_5.mp4',
#     'world5_6.mp4',

#     'world6_1.mp4', x overlap 31
#     'world6_2.mp4',
#     'world6_3.mp4',
#     'world6_4.mp4',
#     'world6_5.mp4', ? overlap 35
#     'world6_6.mp4',

#     'world7_1.mp4', # 37
#     'world7_2.mp4',
#     'world7_3.mp4',
#     'world7_4.mp4', ? overlap 40
#     'world7_5.mp4',
#     'world7_6.mp4', ? overlap 42

#     'world8_1.mp4', # 43
#     'world8_2.mp4',
#     'world8_3.mp4',
#     'world8_4.mp4',
#     'world8_5.mp4',
#     'world8_6.mp4',

#     'world9_1.mp4', # 49
#     'world9_2.mp4', ? overlap 50
#     'world9_3.mp4', o overlap 51
#     'world9_4.mp4',
#     'world9_5.mp4',
#     'world9_6.mp4',

#     'world10_1.mp4',# 55
#     'world10_2.mp4',
#     'world10_3.mp4',
#     'world10_4.mp4',
#     'world10_5.mp4',
#     'world10_6.mp4'，
# ]
