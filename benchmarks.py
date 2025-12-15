Benchmarks = list([

    dict(
        name = 'Forensynth',
        mode = 'wang',
        path = '../datasets/CNNDetection/test',
        classes = ['progan', 'stylegan', 'stylegan2', 'stargan', 'gaugan', 'cyclegan', 'biggan', 'deepfake']
    ),

    dict(
        name = 'UnivFD',
        mode = 'wang', 
        path = '../datasets/diffusion_datasets', 
        classes = ['dalle', 'glide_50_27', 'glide_100_10', 'glide_100_27', 'guided', 'ldm_100', 'ldm_200', 'ldm_200_cfg']
    ),

    dict(
        name = 'GenImage',
        mode = 'GenImage',
        path = '../datasets/GenImage/GenImage',
        classes = ['biggan', 'sdv4', 'vqdm', 'sdv5', 'wukong', 'adm', 'glide', 'midjourney']
    ),

    dict(
        name = 'Synthbuster',
        mode = "Synthbuster",
        path = '../datasets/Synthbuster',
        classes = ['dalle2', 'firefly', 'stable-diffusion-1-4', 'stable-diffusion-xl', 'dalle3', 'glide', 'midjourney-v5', 'stable-diffusion-1-3', 'stable-diffusion-2']
    ),

    dict(
        name='Chameleon',
        mode = 'wang',
        path = '../datasets/',
        classes = ['Chameleon']
    ),

    # dict(
    #     name = 'Commfor_eval',
    #     mode = 'huggingface',
    #     path = '~/.cache/huggingface',
    #     classes = ["commfor"]
    # )


    # dict(
    #     name = 'AIGIBench',
    #     mode = 'wang',
    #     path = '../datasets/AIGIBench/test',
    #     classes = ['BlendFace', 'BLIP', 'CommunityAI', 'DALLE-3', 'E4S', 'FaceSwap', 'FLUX1-dev', 'GLIDE', 'Imagen3',
    #                'Infinite_ID', 'InstantID', 'InSwap', 'IP_Adapter', 'Midjourney', 'PhotoMaker', 'ProGAN', 'R3GAN',
    #                'SD3', 'SDXL', 'SimSwap', 'SocialRF', 'StyleGAN-XL', 'StyleGAN3', 'StyleSwim', 'WFIR']
    # ),
])

