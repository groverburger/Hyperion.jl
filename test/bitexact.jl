# Cross-platform bit-exactness regression — the canonical verification
# that the pipeline produces identical output across Apple Silicon CPU,
# Apple Metal GPU, and NVIDIA CUDA GPU.
#
# Runs on the selected KernelAbstractions backend, computes
# SHA-256 of every stage of the pipeline, and compares against hardcoded
# known-good values. Also decodes the 40 committed PNG fixtures in
# test/fixtures/bitexact/ and verifies pixel equality — so a regression
# is detectable both algorithmically (hash diff) and visually (a failing
# timestamp points you at two reference PNGs you can open and compare).
#
# If this test fails, either:
#   (a) you changed the kernel math intentionally → regenerate baselines
#       (`julia --project tools/bitexact/bitexact_test.jl` on every backend you
#        support, diff with tools/bitexact/diff_bitexact_shas.jl, then copy
#        `data/outputs/bitexact/metal/<ts>/sun.png,dsn.png` to
#        `test/fixtures/bitexact/<ts>_{sun,dsn}.png`, and refresh the
#        KNOWN_GOOD table below), or
#   (b) cross-vendor determinism regressed → see
#       docs/src/reference/cross-vendor-determinism.md for the 14 documented sources
#       of FP divergence and how to audit.
#
# Skipped if the LDEM is not available (flagged as HAS_LDEM in runtests.jl).
# Runtime on CPU backend: ~10-15 minutes for 20 timestamps at 896×512, so
# CPU is only used when explicitly selected.

using Dates
import FileIO

@isdefined(TEST_BACKEND_NAME) || include("test_backend.jl")

const FIXTURES = joinpath(@__DIR__, "fixtures", "bitexact")
const FIXTURES_1M = joinpath(@__DIR__, "fixtures", "bitexact_1m_full_farfield")
const FIXTURES_1M_UP_TO_DATE =
    joinpath(@__DIR__, "fixtures", "bitexact_1m_viper8_barker2023")
include(joinpath(FIXTURES, "known_good_1m_full_farfield.jl"))
include(joinpath(FIXTURES, "known_good_1m_viper8_barker2023.jl"))

const HAS_SHIRLEY_BITEXACT =
    @isdefined(HAS_LDEM) ? HAS_LDEM : Hyp._ldem_ok()
const SHIRLEY_LDEM_PATH =
    @isdefined(LDEM_PATH) ? LDEM_PATH : Hyp._ldem_path()
const HAS_UP_TO_DATE_1M_BITEXACT =
    Hyp._barker_2023_ldem_ok() && Hyp._viper8_nobile_crop_ok()

# Known-good SHAs from the 3-way verified state on julia 1.11.5 +
# KernelAbstractions 0.9.41 + Metal 1.9.3 + CUDA 5.9.0. Every entry
# matched bit-for-bit across Apple CPU ↔ Metal ↔ NVIDIA CUDA.
const KNOWN_GOOD = Dict{String, NamedTuple}(
    "2027-01-05T00-00-00" => (
        sun     = "e98e97f5168eb051b9a9b236f222819633ce7b2ab73ca03c86b2392fb85eb456",
        dsn     = "e7932c0cd9089c149529c5adbdd924d713d269c17ca90dba023ac1a9b19094a9",
        sun_rgb = "87cff154fbd9598e5bd5626e9570cd648d3d50da62b711f9263cfe26cd17b103",
        dsn_rgb = "0ae2096ae7d30ee771af95438eef42f59c971057945c771bbe9f4548e35ba9e1",
        de      = "3f8c4eb8a235b7795144d28860218c1c96a88560b75004b90192deaeaceb55bc",
        azel    = "4a5274c6d349648c5c225eca1d8b90ed18b5dfee5ac4372d927c9d35e1545f5f",
        d_0     = "3329872e586ef14a2ef678a3325af4a70b09b2ea8984b66375845a88084407f7",
        d_1     = "15b2e9c80f3e84b311f988d1b3f2e0e858f9da52694ea14f195930d5427f3be8",
        d_2     = "cff97a680a704a35b99a55ae965ad2f430a562b88ff37f069a46663a7a3770e4",
        d_3     = "f210eecfbe6385fdc36e4fe62f3eefd0da9e76d9939dc49498d33a335f17e5df",
        d_4     = "ca71725bda4a8a3cac3772f5111bf9ff92baa2e29af811f6b96407243ff80ae1",
        d_5     = "8b4e6a6cb49bac8648658db66f674e87cd5a4d9c72797b4b7a23721db6d9ca9b",
        d_6     = "3ee20e3a836bce58ddf71c264616406bf8d60f6cd7ac59d65df13b9c7e5fa31e",
        d_7     = "ea2b0320fd3daca7d345de57790d924d8ba058b33d9d895da2ffd3b7ed29f9a5",
    ),
    "2027-01-22T07-00-00" => (
        sun     = "a09a787f85a6d75f52e194c72a68dc5ecf3a098279b80515e04b743a0476359e",
        dsn     = "28c122ad64b481d9b2405024b52f394011a4d90259fb04f376ee21876fcb6203",
        sun_rgb = "a1c5f07edde63b47e934c51f09d716b9e5541e770b7c173eaeedf0f4996be1b0",
        dsn_rgb = "9b51bcabcb6a8d332395775e1bbdf754fd516015613d3ff7e02ca0ff49af24e2",
        de      = "c50673e1f4554ea45c92d1e7aaf117009273587dc63c722f4ec71d216138209b",
        azel    = "14dd62b80ef8f82307c1b5b1ed1dff1d1cad1def316b570ef025e7b8ff589666",
        d_0     = "c6284625744521113c78f620727139a981ee033b2f983637a34f8cdf1009c96a",
        d_1     = "eafaea37710d968a52b85af71a84375ec8c66c7665fd5f1177f4f1a2fe836f9f",
        d_2     = "e15eeeb623520a2087f2fe727e5f2c8441b023558588a0e74c81a84114d53250",
        d_3     = "1c3bf5fde2d1c7361918a846dc61abb7412ec67d9ed32c7cd109591eef3b11cb",
        d_4     = "cbbb4a357facd6e1f8037ce776bf7875becc783b91a9c01d272a101d0883d917",
        d_5     = "0283f80fa3a63502f2df6024252f8c17a31a74d299949e540be5702e389f20d0",
        d_6     = "9c17a73423f5506d975364e5220b25a98d6129d9962e733c59ffabf41e6db892",
        d_7     = "de036f980b22cd85c783435eeb4357b3c02961081b9383f576a699ae54497715",
    ),
    "2027-02-12T12-00-00" => (
        sun     = "77b404db69d490e45687b4b28e6ece739d8b5a0229174d334761e2be5ea1cfde",
        dsn     = "c558581be7589d96985c053bc2db55549ab13872962a3dc9852643fe97d7f7b4",
        sun_rgb = "faf6185ce0f470b27a9c5f5e63958fe6431aea507ba5de48ffd165fb93062652",
        dsn_rgb = "d95618fcc5d6369f4ac437d1f99fd8a65598e42e882ddbee7f08a4ef0e42d787",
        de      = "542a434bc2be076310b1b85d4873619b62e91fa318bd193aed88475d5e6959b8",
        azel    = "8311ea152764fce0873b612c9f00d6157dbe283910eeae7de409d188e0ab20c4",
        d_0     = "fe3cf417c0c5831785753f6aeb6f1a2b40761d8c5af16274b7f1e5a79adc3e20",
        d_1     = "b5397995acf67aa17098cb66a12150879cabec5b41f6c5fcba13ab8a575fd574",
        d_2     = "4dd71c8e505021c730594d8a2958fcb1f9954d5610fb482ae70547c393ccbd66",
        d_3     = "f57bde996f9fd7c6daecaffd3a1ee083b6997e4da62c17a09620ac354a0408ba",
        d_4     = "58a35b2bba414091930d39103f3a1092f2947cd5b13ae66b76fef7c09832b2f7",
        d_5     = "a870510a6d04a4141ff2e073e60edf929b9cd6f0835e17772c171a31489be75d",
        d_6     = "4f5e638853a9fc3cf2e6ee3e5a4a06b9907ef080f8774a3291866b5a1e6d729c",
        d_7     = "e690d16fbfb637ee30a2da4e86768b26e2bbb89f3cf20e83f95c50cae0483fcd",
    ),
    "2027-02-28T04-00-00" => (
        sun     = "17c6acacfbe899826fd097ce07f0018a9ab2850bfd1246470cdd3f57b2b3737d",
        dsn     = "d639162f5b79c3eb102ad5fa75a36a782ace9a6ddfe915dfa72433a70b9809fb",
        sun_rgb = "5a0f230090bc823a438ad33c83f0d170cd8b8b97ce0e79ca98c3e212537cbe50",
        dsn_rgb = "c4b016c8ff3827a8a07a66f6d28b369ecc7bbaab2ccd46f9a8964942dee43179",
        de      = "9eb1d45cd7afc6102aa3afb5cadc68779fb8ad11f5d4422e2388e8f0e346e9dc",
        azel    = "7e6ecfc47cb8186e6e02c322febd179c7659dd2e6e14489db3c0b81a02e85469",
        d_0     = "9ce7cf8382a09b5e5dccc001f296f3af163976b48db02b62287911e90ac7be8d",
        d_1     = "653a0ab02eee50b47428b9c450cca7d302262423c5ceaa136b25d043c8ee2984",
        d_2     = "77ad57e228dcff27776a7c8e562aed9831be2180e1ab8ab66dc7855f0008a80e",
        d_3     = "e999cce07686786debd809efd7275e7f39ef3d47216dbec5561e8491ac549897",
        d_4     = "902433388f19c66b9c0c7a69f32d047438889bd8cca962ad42f04aa19bfe0561",
        d_5     = "a7609161110b552d691a10921205eb6763b30b350a839bd4d4c5cc7cf8d70b82",
        d_6     = "ef3a2d7089d947c58e54770b470bd956ae3751fc867b9b7338040960844d3dbb",
        d_7     = "575ec275734f6d321c5709319a9cd76867502493e6907b92e5845d8cd0d6eaa3",
    ),
    "2027-03-20T06-00-00" => (
        sun     = "f82cb44686a766e45e8295012bfd479d017b53bb09a020f5e9609f3ab848a959",
        dsn     = "982ce9ce7e8f4a18d1ce170bf07e2aace056090c3fa3c2fe6ceae764c80f8d1a",
        sun_rgb = "f531814aafde9e95e6153dc6572dbb88b992684e1ef6b2fbb9905ea08f66c96e",
        dsn_rgb = "a20f3722138828bbdbf48509cb383314c01773dfa4f7f3c10cffb9b462d08f55",
        de      = "d15cfdb2501b904069973de8421bf31601420d53aa7b60c5f88904876d282a36",
        azel    = "fe4bc313fe7875e8b3fc845798bd55c4efd017593e407caab2aba33234882c3e",
        d_0     = "8ee47876e1df3b12c52b983f4ce9e545621b76e09e31689be823f101d76d403e",
        d_1     = "de5792b169691c16bb7c4b1dff731e726e15222c466094ad500c3ade79c2aaf8",
        d_2     = "0830b8bfa87469af2c3521e3aa56c1e537f62b4f5040541c1782e8949e44348f",
        d_3     = "c4d49de452b1ce6a54f5c3a7819738f168fc1f09bbd208aee386b7513f3ee04d",
        d_4     = "77f2b13a3dd7a0cec04f207972b72eb1c45937cb62ef0da9bbb54d413653139b",
        d_5     = "13eee190f0eaf887821f7680ac757632dcb313fcf7c686e87a2e4b171d9ace16",
        d_6     = "cc435f797bc44afe4c75e7cc93825a103b7aad4eaf153bdf90bdec2da0c66a47",
        d_7     = "b2608452a6b5369d6d2a18e973f36f93d1e4ec97631402edc19939c3d99fbc56",
    ),
    "2027-04-08T18-00-00" => (
        sun     = "37938c488a9a2139c1f2c5f7021bfa94e212d886274f1dce845c0197cf4faf75",
        dsn     = "df946afd303fe761f6318769a6832a5d1687ce7526c08bdf0aa89419631856ab",
        sun_rgb = "ddb4dec5bac452ea8fc6e0f1e721bbfd25539ef9b34559e6474c6dde8fff5391",
        dsn_rgb = "6d8496b582d827d970de1bc9488f3c8cf7d50a714fbe672ba329f8486037227b",
        de      = "bf4de152993060db29837e4d054e73ca5cc9b2e94b2001fd5ad239bdc9efb4a5",
        azel    = "4ee715efd4f7f4a5bcdc01e04e68896399126a9b6cff1a24ad8e46ca65746649",
        d_0     = "e9d4b30564930b57ee3652ecd8b3f602d2e813f6060b8e3ff1b75d270e977240",
        d_1     = "928707fb3fd292a2f7a21037c838322b8045bdb3f8210a2d55d4fa3ccdb502f6",
        d_2     = "f1be5e1d9f1e29e26141b13570010db9d0b2e8388163d3fc539cb1836a6d5f49",
        d_3     = "0fb8752bc2dceef01c9551bc4e7731c65e8f2fe22e8a21158b5eb442fa93e8f6",
        d_4     = "f24f898ca51fa6a4da6fddfaa668810aed4a594d4f1165bb91d39b6e8db3b874",
        d_5     = "f74a5dede5609615481a6fd31e6d7a27c93a06f43f106a2eeca4ec3e7aadb9fb",
        d_6     = "04b2252f5857f26e64bf8acfe18d4724c7c998f432efcf3a89ecceb5598e4049",
        d_7     = "620aa6a7e7a52b3e109ce1550bf86b79673ec658a642a29b38b92c91745d7941",
    ),
    "2027-05-15T00-00-00" => (
        sun     = "f74ac9ba129d46957c837afee44e5ba2446997d5dfc7535e086c713b68e4506d",
        dsn     = "57f9ef0fbc7cc33a6a702faf76b3e5549da944d909c27e3e31f5412c2d31641f",
        sun_rgb = "c8e3656b21ce0a734dbe3ef27010e021e1749273d90fe90f1b3517b16d591e21",
        dsn_rgb = "bca4896d6db74fb26046130d0677fd65724d107a627709173384ef1ab9624c8e",
        de      = "fe1f33546d8c40a4a878e1b64a4632931354e5b3816d00fe0db330bb45a1860b",
        azel    = "d451558285bdcba6384b71432f890bdd3b12ce7281f24b9d9055937266325af4",
        d_0     = "1969280ecec66522d44d4fa2d285c8fc33d2b8d3c6f14a825efacd6de20c2bf6",
        d_1     = "3a9b838d5608b099034131d8806129105e3936a29d5ad3953a190c1689540e2b",
        d_2     = "1ee0af9ad56c6621f05acde686985f8120563e324ab1d798aacb15d9454c06cc",
        d_3     = "44018e0b7d07d6ac40f41b1bf2ab50b8b3a162bf30706ac21ec9e25af3fd24eb",
        d_4     = "1d39a4123ba21ac6f871e9bdd5e6bbf95d23cbd5def42cc2ea40e6aaf2155d59",
        d_5     = "c72aaf7d25ee7ab72e014b7121e59555e96f1c6d7b3132272076e297b03b4dc0",
        d_6     = "cc2830d049a6a2ff420cd177dbe9a5a6d82afa19038a49186ea4a2776b904c77",
        d_7     = "bf2c668d53e5734cfdb4dcb0e6f4b8c87f6c7ed0e043f622ae51dd4da58482b2",
    ),
    "2027-05-24T19-00-00" => (
        sun     = "cd7ba6bcdffde5e51c7e9d30f4a206daad07f29024029cd8d89a9dd626ab3866",
        dsn     = "f49b35ce46e3391ff65810dea666db6b59e921bc684e54abceeb4d43ef0363b7",
        sun_rgb = "17411a05bba857f87aeff4f98ce05e02330060f24ed909010d67a028c7455d1c",
        dsn_rgb = "9be95747fb634fb7a0104615aa5dc1f05eab2f61b7caa138724fff2a2982d84d",
        de      = "626146e6a42b9383086284918fc85c16d9c0fe0a2fbadf38c0c265e7f97c714d",
        azel    = "0eb19dce2dafd99fef7454efe14797ae8af09d120cdbdbb6036c7f951c560f49",
        d_0     = "e0f2ee264fc155998c779ff17e6187148381677f3628f3c473c83ab855c8060f",
        d_1     = "00eedd7c87265a50441b106ef894c42e9792d6739e4514dfcc92e24edd3a0005",
        d_2     = "04ca9a25fefc3b55fb9bc6f8adcb4fb8e05d32d1269063384f8f21a2b5e4fc91",
        d_3     = "ddb6f7a1ce64f781f647057ca5b169362f7edac53d7ac926d6143bda94677ccc",
        d_4     = "9d90d8341c09cae43a9442e5498f3fba7f6e43866a6e8816252cf3183008829f",
        d_5     = "cb034c574f2521bd3e264ea8783f7b0f24fb93ccabffd70410df06aae23a8ea9",
        d_6     = "fa354e76aae9d036159ad231e8b9b5cbffb7e2505ef37dcdd7b73687d005dd6c",
        d_7     = "718a437cd40068a64b26858179917157db6dd740055745021cfb7ca01c14c602",
    ),
    "2027-06-01T00-00-00" => (
        sun     = "183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1",
        dsn     = "561de8abc4b7fbe1b71a0b7bd2bc2d3b628ea7fb54b88c22e813e115fd5c8b94",
        sun_rgb = "9794838a8913f06045b1b48909d103d4208f6a8050845b796cdcc20e5600452f",
        dsn_rgb = "148eb4d6ae353ecb8ac0fb1e8c7b44d10eb5e0e2d836073d4dc5449e4607b189",
        de      = "d27abbd27d30a386342e7f305c2e7d476000999e3429084edef24a888cf3575f",
        azel    = "74991825f0839f7b115d3c3c693c360414a01e81308de4cd6830787d43fa4366",
        d_0     = "1fae98367646e4d3f53f30c3a3324c9e14cb1b75965ea8f26fede502eedf9d79",
        d_1     = "761769fa4fed4df10dc6b1b26f26c6fc335111522c81c8e872b0b4fbbb119fe0",
        d_2     = "62bd793b5a911a90af5b3dd3f0e944c92a51a31df89ec308d5e3be2471fd8984",
        d_3     = "4f4982ec064445ee05cda1790ec7f8b7f978e19e17bd183b540ed70b4f43c539",
        d_4     = "550e17c0ab6d21cebf8553e0142c274c82d8940c01a4330e8b7664b0091cf342",
        d_5     = "928714c6e09d9e41e494bf6849429ace32585dcddf1d3a96e336406dbbe523e2",
        d_6     = "cb4f7d13fbb3fedc774f9863503e69a70a80de85076b1841d8b2c673801ac2ac",
        d_7     = "365e72ea9a9184dc1963d1902710201eb8ba68f1fb26ca0cb2b04dc03236cd4a",
    ),
    "2027-06-21T12-00-00" => (
        sun     = "59061b90b6215eea996751579e6124438e3887f3fe060c7f0b49e5514fd014cd",
        dsn     = "c2c79f0e501bb23851761ddf2c94cfc4262b9b5efb1f400d500775541c505b77",
        sun_rgb = "2e8f6a2f02dfc19e1c2e1afd2b973738e73e70bd39584a2b5d605e526c7e1725",
        dsn_rgb = "03ccaaa94efcd82ef5c0727ba619bb0d61960999d2e4d3fe49afb11fd2701317",
        de      = "827ed798b37fc542bb46aba926bf9860a4da4b6a81b4e9b2367a1daf95803650",
        azel    = "c71d434e94fbac0fac12dd549d76c52c80daeb59890d6715b6c70a91ba1bf3fb",
        d_0     = "4c0d410b08a792d7a5e7fb897049e66819a7aaf2fbca48fcd64d9ba0604478e0",
        d_1     = "42b0c277610738c12279a4abd077c6c945820a77569af04a499f365aa5a11826",
        d_2     = "250a14e2112ced5961779af309447b92f379128b057996890d779c8a4d28449d",
        d_3     = "8503deed722f284d4b5a1ee669219343bad8c5adc3fe5507256637d4f8c95031",
        d_4     = "4b80cad17e1f3f0f364578b561f3bca3701a96f70532b0210c7a30cc4ac8b0d6",
        d_5     = "3c14c04d9ceb773dff75b721391ada6c74a86aad3d46e73c6e02a8689dc86a88",
        d_6     = "388e8e74d14de3e2d9430d6573c57696b1f1fba95547b8b9e11dfe6fa8b16b05",
        d_7     = "f17a59ab372b6d55b758d4d7d1ea43bb1abee2cfea251b5306fe6f6187d521cd",
    ),
    "2027-06-23T00-00-00" => (
        sun     = "8d755e306ab1816d792160c9913aac390a37f6c468e0e945fa4ec1af721e552c",
        dsn     = "aa7e3dcf80539a0bdcee9369fa2f6b38f5581e16b2a3075f290f7e638c9b5f1e",
        sun_rgb = "b747b1a788625635537bf128d727630efd59a861838bd3778eb2c495cb75bb45",
        dsn_rgb = "c98379d6dc3ab926f92cf03bdc97ba1ff3f091a3c406d6fb54184a46e5fcc105",
        de      = "5e3ac7a1af91519a186d368e6411f572a743f40d73765090a7858109265d8288",
        azel    = "875ed793e2712b55d75626ddc42eef42c7848da214cb7a3732ae734df64b4933",
        d_0     = "2ce3bdbc9b5f5ca500262f333757a1f73d20b54909b005269c2d3bb9d3087b90",
        d_1     = "8e0f11eff4005831446f7f66399b633f59050758d8c88ea5d7882a6da88aca43",
        d_2     = "51ca24f16e4c728c53646fbf092d6eb55e1d1bfd90d07d8d44ac771244c75130",
        d_3     = "d531b9dd2d1ca77aed2dd3813295c6a7428a364803c9e9d5a152efb821cc1ccc",
        d_4     = "c014c5304984378ef50f7bc25aa74b3262a99830253f75f000e05d6495a311b3",
        d_5     = "639f7dedc27ca7a35f0de805f58ba661ad15508da565a1e2fac88e6031bbef6d",
        d_6     = "179347ac55740aec4417782c7231552b2080da971722d47ac7693bbc70b8b4b8",
        d_7     = "ef48a2fcc91dbb02b5578823dbbcd5cc3150ee2648bda9e1fb8eac1f372d08e7",
    ),
    "2027-07-04T03-00-00" => (
        sun     = "466c8268d9983a1899db028ac348842233324ef2e886ee709cf612965ea5a9a1",
        dsn     = "59af54ee982272bb269d3a32e4fa0ed1dd54a303f0694e312e89d4571c726cb8",
        sun_rgb = "669d188767be5756684a0cebb1432dbc7cb7f9458353724dbb0b548f088c2dcc",
        dsn_rgb = "d632c4c77c6c4edfd87346b056eb09376ac7452e1a43af7e6f50dd314ee57c79",
        de      = "6bd2f329ec893ee490483c3adb90e03998f854d76c42f2251b36f75f5d961f56",
        azel    = "da0f32c07cfcf2e733763a4e1b2fffef4106b8c82988fae02e87c9a46e0a1616",
        d_0     = "38ba73d636bd3d1ac2b863484e618d17f506c3b6e7c42ddccadbc2adfcaccb9e",
        d_1     = "a1470403db0b5bafc64b9eaa97b3c98266ea62c5e53cd902165c3ba1fcf39310",
        d_2     = "9f496bb30a2fe0a0c913360e840dbff77ac1e00c21d839562e94c046e08cafbe",
        d_3     = "92addd49a25aa5238f6484538a29d2872805f1aace483b24698bf1c8c634cc3a",
        d_4     = "15d2e296253256f6825343bf96e5f85b5da8c1571a7c3ba85fd111a40bf1ff17",
        d_5     = "5e04f7e5607ee66f409c7a4a49283aebdd5788fddbb0103cba814c82f90c04c4",
        d_6     = "30e9ed795f4a7fea3df212b09292da5b7db1d38444813edfbfc113395f6abfa2",
        d_7     = "b2fbe741684bc4f80931fd658760f93c0b40466e896dec9f38a761b515ac91ed",
    ),
    "2027-07-16T08-00-00" => (
        sun     = "4fe37f794cd2e37fde7f93073551d2cdbb290999c867e63e6ba640a667339042",
        dsn     = "1957b6bdc365897b1089595420c593457149a19a4166ab2c855d193839dce620",
        sun_rgb = "e32acc6391006df841cba3e8140735f8c7de7c730f57c3b037ba9c2a980ea0ae",
        dsn_rgb = "614633916e38564ed55d25d8f78f675e34ac3562fe6909adf168393c24ce9ff2",
        de      = "aea2af027d2d4f537ea9e2cbf02722f374c174e8b26c8ec419b28145a5fb71d0",
        azel    = "8f43e54452cffbf87282f81825fa36509d793a981e9579906696cbe0773507e2",
        d_0     = "0a214fd0133bf6922f150cbe91a6a38cf10774fc3a5a0e76632ddf4ee4af59d1",
        d_1     = "fe1a4312678e3b84441d7d784d80902daba6bf5938f8b9aedae55fba36d3d147",
        d_2     = "4ff99ca68fd387b48ab5d9d3279a01c8938d138f8b1a3ffef900d0119dfdd0c3",
        d_3     = "431a0a23f0f93719f397a8c59ef5a0e19da3c1fd0d42ab4314039a5e60c7c199",
        d_4     = "4b807c70a4a01d864b6ce7ac3fc46eaf244bd5a6a316da41093a8f3d3c2deda6",
        d_5     = "aed08cfb8b41c58ece4c6abb80eb1ba4b91e3a13748e0516f461b334058001b0",
        d_6     = "0ed5887c4cd53bbb498537d5287c734b6b2e4723c707afb5ba1c9cc37848e451",
        d_7     = "c86f7f7fdaf3b3306ae55d927b50e37dc2a07ffab3bdbd287b10db54c509f922",
    ),
    "2027-08-05T11-00-00" => (
        sun     = "394626486e7b76035a41639775e60ad8783e408b813578598a5231e05a7eca5a",
        dsn     = "209545d9a03c36600e27ef58196bed7803682befe9302bf31e89b12fa1b93290",
        sun_rgb = "b4f40d5c8b42fa03a03d633debd909b8e47ceb0ebc3078150731ba408cbc1526",
        dsn_rgb = "33a8fa3337416ff96aa4133e364672d5f8e5440a49173e0888fa1f8b23278711",
        de      = "4f7d881b064557df35e4757e14017fa81dcceba7f28be196601c00b45e30c2dd",
        azel    = "b2b62d632f0df5856bf570effdee045b928c1b1e8f3c4d4eb41ab954e8a249e0",
        d_0     = "aa38a6f63da4fe78ffd5fa348980fe2edfeeb9775299dae4a6721b733c358cfa",
        d_1     = "7687ba22897d7b36498dbae7d38293615fd3852c841f0ae3fecbb6561f4f1c3e",
        d_2     = "c1c5974934b5ec0117a90d67ecdad8c5b6a3fa5da55ab4dad3e54c000ed8e6a7",
        d_3     = "0addbb7c5ae6fd1bc856b160293b21cb6ad32af552d7b58f4dbb97d4754075aa",
        d_4     = "e99d58816d84e1339c3c3804d6d3d79604c70b5bab4355c6a2d2e6e09ceba082",
        d_5     = "1e098d49c5bb9efad5514b9664a810a4f8db5990120f96f10c7417fa31a5c556",
        d_6     = "c2fd91442cc750fbb295a6f0a263e9c15c09b1ae6fcbd08bd929b4164cd75c91",
        d_7     = "f21a6de621f7859284f7b1b69c4516114672b5ed7b06591fe31d00524e17841b",
    ),
    "2027-08-20T15-00-00" => (
        sun     = "589a15254f0fcdd88ea7bb45a58c108b1b5467fba9168164bf2a7559bc40ef6a",
        dsn     = "6b4524ab766f34136db08fb69a8b1db41713fd7cb02482e0ef485e6f086bf612",
        sun_rgb = "42dc90f3634ef61c43ebcdaffc224264219a04445eacd753765012de511fb741",
        dsn_rgb = "60ef06d3696cd2bbfd55e609696a3ccf63fcac2e54e9839a0d8ddb8fd5d54887",
        de      = "ab26aad20164b8de45bdbb4e020672343e6787aa99c6f8c568b453e26193d4a9",
        azel    = "53c39179259c434ed87584df805e4fa94d1bb5240a0c0d9f34dd75629ee11878",
        d_0     = "ed0c512ff03d72d82a1d12abbd2f2b5235e248f3ad272e62e2ebf1268a769601",
        d_1     = "4cd759644be4e08a96d758db329cd211961206384449d0a44618bf93583f5864",
        d_2     = "4f8a0264a7c43ec10090816c1d6890c70e0e81e41a7510f321e36840deeb5628",
        d_3     = "a7e05626a1b7bb57d2239d4c09d87223ef560b06b85cfff6528eef51b5131dbd",
        d_4     = "167cab256dd90c4c6a7638b62326a244a6b3f11bd40d700c3edef18d6f6149b6",
        d_5     = "f9500d8e95be03f546788d7aea12d88c10500b3c317953b0d57855907af80bd4",
        d_6     = "171da1c1312b1451aa7bc18d928b56d53424402b2b7861b01e198fcceda992ff",
        d_7     = "f46fc0d3265963f0182f5c8e050e4680632e2776dd81465e48e53f761821f2e8",
    ),
    "2027-09-10T00-00-00" => (
        sun     = "4deefacd0efcdbf26b450bbe66adfae1415f139290904298aca65fd043e75607",
        dsn     = "8931fcf78cc6fa032c83945a07260d30d7a0e9e1a80c177e58e7b9b03871be80",
        sun_rgb = "be101aab1333fe20fa7d512ca3e3bd86722d31faa969875c91e5bb322f4bc8f6",
        dsn_rgb = "aabcea1cd9a8173c2a39b936c821fd18414102c54e0da7053797725b4bd4b3d5",
        de      = "2c65566fdc12e3ffc4ba139bc8e709e477687981e12583c3e84fc342c153d0c9",
        azel    = "f83b864e6605aa9bb52dfadc67e9345fcf749608f0628bb7b8f0aaac1a9f8bea",
        d_0     = "b84fac861a8b3446fe7921cbf40daf291be5234b39a430cef4e04b87d9b407f9",
        d_1     = "06d764afc541827f5bc6c0a1c5d56a1f50d36da64a84dc3a1555a0e75e911de2",
        d_2     = "03c85106dfd3afa1a0eae005af935ca5d3379157ea5195fc9f70b4e6b32cbdef",
        d_3     = "842c5eba0dc9334baacfb99b759c645df75ac6ec72b9157fda111ef11aafd5b1",
        d_4     = "50bdd0d7031d2f2a835ff0df5a03a48416b226213e64d8bbd6c4e60f70b07216",
        d_5     = "3cfb82aa5df140de7a1458e926de41b45c7103cf103b9136454d355addef749f",
        d_6     = "f4cc0cbcb6dbe7603e30a619df82105f19a0282a06df76427bb3286c3d10806a",
        d_7     = "3873bbd05e3243206af873c73583c33d8c8b8e2075137be4e778d525f0eddb91",
    ),
    "2027-10-05T21-00-00" => (
        sun     = "e2d97c0448254f92abae288b1ec9e9ae90fdd9aad5817be09c7489fbb422b8c2",
        dsn     = "3cdcac141766113804c087eb6c2d0bcadf15725e0d5d1d65ad6b649ec4373001",
        sun_rgb = "20a0fdcaf2baefad2272842f16c3e499c454f1424368c3763cb6c0978c45e183",
        dsn_rgb = "9e546570ee5ae124c07b2ea0921bdd9e548c348b6834c60f3e2d5c3c4da4530b",
        de      = "5688f60714a02b67b7b3d08a861a95f1afc20ee34a0371e65ce0c0f1402eb8a3",
        azel    = "1fc0736570b9c1eb067c6aba2a2aeecee8f1377939b83e30b82720acc078c9b2",
        d_0     = "87a639116af5686fdde421ad31e6816baea9c11b0b23fb481ba926ba31d20884",
        d_1     = "3270d6bd95106aa36513c5604bd666957aadef754d57e03b612c94a54453a4ee",
        d_2     = "4db7e241ec4b8070fc673f6f99cfd87caaf7882472dc7e86f204a285e6c40427",
        d_3     = "77efda4f201c470ce5e0b7349ed3add8136fb9260d9674c4a1b22067e2830c91",
        d_4     = "0e740fedba58ad62b6a5bb3fae02fec4e3c5034c573b7add7982b4617c3a41b3",
        d_5     = "6d7c8044c3261a32917682d3fa9d6444815cb595e65af3e8f610cd1f7fd5fe4a",
        d_6     = "b0f9231ba9aaabf4181abb53388fa683eb59c28897754279b5c700fc96b19e1e",
        d_7     = "62ffad2e4f94cab46638f99750ac26303cfdfbe1bd0239f445ab6adfea26d105",
    ),
    "2027-10-27T14-00-00" => (
        sun     = "745ddde64edb9e0f8f62ec2c543a0b7e70c7acf0e05402ffaaf90d325c940fec",
        dsn     = "9fcece9419c37d282aefc6eeb494ebf656aa3ef4d5a4d14f61ad2ba999ed5b21",
        sun_rgb = "f03156ff903d1c4ba5bbfca5978d414add0842f1fdb840f12c37b963a42c2b4f",
        dsn_rgb = "1210289f05bbc26fc1c75aa937818beacab40b5f079bbf5f0fd073915960a24d",
        de      = "d2925eb3737e3e7866128b44f97135112bafe60c1a6ecf1c1f9c62761d99b251",
        azel    = "3a0e0382cb9b37d919a1aab33a792c631a625b84cfe402aa2b2b80b0c6c647be",
        d_0     = "8a98a74fa0cda7d7a882ad5dba960324e8b3309178f4b67ffc3670c604cb380b",
        d_1     = "fff672efb425d75e415112143b083669371832ac554126aa07b1da375f78cf96",
        d_2     = "8dacf2970102fea0fd8c7d9bbd6b0efd5c141c671aecd988c88491dc90b1451e",
        d_3     = "3bd8efac746ddeec030987a8d0071297c31912f6ece8845e5c00c41cd66b4fcb",
        d_4     = "e7d0306785aa265400e6a949a07bb04843c20a151bd73294432b56356e624407",
        d_5     = "a28e0d10ab0b9073f16e84eec5fe0860c0f663176a126a4a97b282fbafeec1f3",
        d_6     = "e74daf93d6f6227b81f24034fa2d5408aa435d9af3f2a4f036d06a96f387929e",
        d_7     = "858692b00f23ce4da8d41eb89d6ae55e4feaa1856063cd1e7300ed44fba40595",
    ),
    "2027-11-11T12-00-00" => (
        sun     = "88eb35a0253873178adff827a3e7b24380e9057670971bcc3e401136a78060fe",
        dsn     = "b24a87e0099f9045cb2bcc40221449359a1b3e8f216ab27be1899e8ec1b23f60",
        sun_rgb = "848886c3da81468b95c20b55118f2988fdb59271c010bc1184265cd9b9a784b2",
        dsn_rgb = "d685ab518d68c719fe74d90948ae64d565a7c9eeffca8b99a3a4ec084a125773",
        de      = "c0585255439e1e6156c192a116d0d2a4bb68b1854144535abaaf91520140ace9",
        azel    = "73da292d350ad6b4e91f42b30e24888df3ade8c839c8da391688f982e70d98ac",
        d_0     = "9b62608d581dd57cee65b3ea4258402da13732cb2b45d5881ced6441036dee14",
        d_1     = "9f7c38ee12239e91df73502235dc8b3cc8a88ffffcbb0e518ced49038f4cbf31",
        d_2     = "918e22a98547d29b02fb741792319da4b855ec200a2ec9ea950c58e4c3b8c5a9",
        d_3     = "1c98ddd789cb114c63a12a53aadba764555b3ca60c558b3cf404c3cb710f6144",
        d_4     = "3f1598a88112751119fd72af86adc2bb47c8de317f2ed4f65c79bbb52731dc9c",
        d_5     = "6393ec25b6763c641bc3030537af50085501255b0432ccfc9dfff1376271afd3",
        d_6     = "04b50644dcb8dcba4567db3e5ccd01661fd05c27ee5a13407cd25831df75230f",
        d_7     = "472268b1ae05cd004bc9cbcfef733035432e2ba3c4ecdf15ea3584d1d4c17d2e",
    ),
    "2027-12-21T00-00-00" => (
        sun     = "c0a301523c5958914fc353fbb8d9a2ce339b72c87dffd1c42e194c2b455b5d65",
        dsn     = "913c500d680105de8d69a1dc1da81ce30876761a987ebfe01d50a8146a89f116",
        sun_rgb = "d852fd28d311e6e014f53580cd1cebc2b4c3b903b13aaf80aba9411a99ff74e7",
        dsn_rgb = "87d30e586a292929c28cfa5014bc9e41645252c0d17c0e3b976c0410be13baac",
        de      = "997627525f6ec42d4bec56acb77bf7c6b97aa0829f3be41a36060576c407537c",
        azel    = "cbc62238b56e338f4d69513c377bea9cf99a0006d70be4ea4929397583286486",
        d_0     = "87c267cd48b119816eabae9099d8a755510ce7dd58e2ba61374fec5efcad33ca",
        d_1     = "810f8cab9d371eaad693860b0a8e8922aabe23769616be7cd910cafdca4bbdb8",
        d_2     = "2cb5d0c606063d018ea86fa926d4c2675ead07474e9f5660fea23316b160296b",
        d_3     = "950f793871c7ef463046ffab0e147fd4aca52cb0edbc3b85421a29af083edd98",
        d_4     = "ce287e340be2c1430a5dc2c55807eb8c74b0e6b7c5af79b3cabb593ae4a6ace0",
        d_5     = "edb6e99c6e59e8543e2a599f6d80fec4084f40f48f10d8803da5cff982636199",
        d_6     = "fdf193b6c6b2df2f1e1898c29d1305668cbd2278e397da4f37938f480ec7bddf",
        d_7     = "ea8a0c767feb7f4453565ba14caec10685fd9a21bb80b6b4b408460d1ce81268",
    ),
)

# Palette application — must match tools/bitexact/bitexact_test.jl's version
# exactly, since the sun_rgb / dsn_rgb SHAs in KNOWN_GOOD were computed
# from this layout.
function _palette_apply(data::Matrix{UInt8}, palette::Matrix{UInt8})
    H, W = size(data)
    rgb = Array{UInt8, 3}(undef, 3, H, W)
    @inbounds for c in 1:W, r in 1:H
        idx = data[r, c] + 1
        rgb[1, r, c] = palette[idx, 1]
        rgb[2, r, c] = palette[idx, 2]
        rgb[3, r, c] = palette[idx, 3]
    end
    return rgb
end

# PNG → 3×H×W UInt8, matching `_palette_apply` memory layout so we can
# compare equality directly against the kernel's RGB output.
function _load_png_rgb(path::AbstractString)
    img = FileIO.load(path)      # Matrix{RGB{N0f8}}, column-major
    H, W = size(img)
    rgb = Array{UInt8, 3}(undef, 3, H, W)
    @inbounds for c in 1:W, r in 1:H
        px = img[r, c]
        rgb[1, r, c] = reinterpret(UInt8, px.r)
        rgb[2, r, c] = reinterpret(UInt8, px.g)
        rgb[3, r, c] = reinterpret(UInt8, px.b)
    end
    return rgb
end

if TEST_BACKEND_NAME == "none"
    @warn "No render backend selected — skipping cross-platform bit-exactness test"
else

Hyp.init_spice(joinpath(PROJECT_ROOT, "kernels"))

if HAS_SHIRLEY_BITEXACT
    ldem = Hyp.load_ldem(SHIRLEY_LDEM_PATH)
    max_mm, min_mm = Hyp.build_ldem_mipmaps_minmax(ldem.data)
    origin_r, origin_c = 8960, 18432
    H, W = 512, 896

    for (tag, expected) in sort(collect(KNOWN_GOOD), by = first)
        @testset "Cross-platform bit-exactness $tag ($(TEST_BACKEND_NAME))" begin
            dt = DateTime(tag, dateformat"yyyy-mm-ddTHH-MM-SS")
            et = Hyp.datetime_to_et(dt)
            sun_t   = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,   et))
            earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

            # Run kernel on the selected machine-appropriate backend. Output
            # matches CPU, Metal, and CUDA bit-for-bit per the verified
            # 3-way invariant.
            sun, dsn, de, sun_rays = Hyp.generate_live_shadow_frame_gpu(
                ldem.data, origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
                max_mipmaps = max_mm, min_mipmaps = min_mm,
                backend = TEST_BACKEND, DeviceArray = TEST_DEVICE_ARRAY)

            # CPU precompute buffer — its SHA is part of the audit trail
            # because a drift in CPU math would silently corrupt GPU input.
            sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el, sun_tan, dsn_tan =
                Hyp._precompute_azel(ldem.data, origin_r, origin_c, H, W,
                                    sun_t, earth_t, Float32(0.0))
            azel_bytes = vcat(vec(sun_rc), vec(sun_rs), vec(sun_el),
                              vec(earth_rc), vec(earth_rs), vec(earth_el),
                              vec(sun_tan), vec(dsn_tan))

            # Kernel raw outputs
            @test sha256_bytes(sun) == expected.sun
            @test sha256_bytes(dsn) == expected.dsn
            @test bytes2hex(SHA.sha256(reinterpret(UInt8, azel_bytes))) == expected.azel
            @test sha256_bytes(de) == expected.de
            for k in 1:8
                @test sha256_bytes(view(sun_rays, :, :, k)) ==
                      getfield(expected, Symbol("d_$(k-1)"))
            end

            # Palette-applied RGB (the user-visible content)
            sun_rgb = _palette_apply(sun, Hyp.SUN_PALETTE)
            dsn_rgb = _palette_apply(dsn, Hyp.DSN_PALETTE)
            @test sha256_bytes(sun_rgb) == expected.sun_rgb
            @test sha256_bytes(dsn_rgb) == expected.dsn_rgb

            # PNG fixture comparison — decode the committed reference and
            # verify pixel-for-pixel equality. On a mismatch, the reference
            # PNG is directly openable for visual inspection.
            ref_sun = _load_png_rgb(joinpath(FIXTURES, "$(tag)_sun.png"))
            ref_dsn = _load_png_rgb(joinpath(FIXTURES, "$(tag)_dsn.png"))
            @test ref_sun == sun_rgb
            @test ref_dsn == dsn_rgb
        end
    end

    if !isfile(Hyp._nobile_1m_path())
        @warn "Nobile 1m site DEM not found; skipping full 1m + Shirley farfield bit-exactness test." path=Hyp._nobile_1m_path()
    else
        site_path = Hyp.require_nobile_1m_tif!()
        site = Hyp.load_site_dem_f32(site_path)
        far = Hyp.PolarStereoTerrain(ldem.data;
            max_mipmaps = max_mm,
            min_mipmaps = min_mm,
            elev_scale_to_m = ldem.elev_scale_to_m)
        stack_1m = Hyp.TerrainStack(
            Hyp.SiteTerrain(site; window = (0, 0, site.H, site.W)),
            far)

        @testset "1m full site + Shirley 20m farfield dimensions" begin
            @test site.H == 4096
            @test site.W == 4992
            @test site.pixel_size_m ≈ 1.0
        end

        for (tag, expected) in sort(collect(KNOWN_GOOD_1M_FULL_FARFIELD), by = first)
            @testset "1m full + 20m farfield bit-exactness $tag ($(TEST_BACKEND_NAME))" begin
                dt = DateTime(tag, dateformat"yyyy-mm-ddTHH-MM-SS")
                et = Hyp.datetime_to_et(dt)
                sun_t   = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,   et))
                earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

                sun, dsn, de, sun_rays = Hyp.render_terrain_stack_gpu(
                    stack_1m, sun_t, earth_t, 0.0;
                    backend = TEST_BACKEND,
                    DeviceArray = TEST_DEVICE_ARRAY)

                @test sha256_bytes(sun) == expected.sun
                @test sha256_bytes(dsn) == expected.dsn
                @test sha256_bytes(de)  == expected.de
                for k in 1:8
                    @test sha256_bytes(view(sun_rays, :, :, k)) ==
                          getfield(expected, Symbol("d_$(k-1)"))
                end

                sun_rgb = _palette_apply(sun, Hyp.SUN_PALETTE)
                dsn_rgb = _palette_apply(dsn, Hyp.DSN_PALETTE)
                @test sha256_bytes(sun_rgb) == expected.sun_rgb
                @test sha256_bytes(dsn_rgb) == expected.dsn_rgb

                ref_sun = _load_png_rgb(joinpath(FIXTURES_1M, "$(tag)_sun.png"))
                ref_dsn = _load_png_rgb(joinpath(FIXTURES_1M, "$(tag)_dsn.png"))
                @test ref_sun == sun_rgb
                @test ref_dsn == dsn_rgb
            end
        end
    end
else
    @warn "Shirley LDEM not available — skipping legacy bit-exactness tests"
end

if HAS_UP_TO_DATE_1M_BITEXACT
    ldem_2023 = Hyp.load_ldem(Hyp.require_barker_2023_ldem!())
    ldem_2023_max, ldem_2023_min = Hyp.build_ldem_mipmaps_minmax(ldem_2023.data)

    site_8 = Hyp.load_site_dem_f32(Hyp.require_viper8_nobile_crop_tif!())
    far_2023 = Hyp.PolarStereoTerrain(ldem_2023.data;
        max_mipmaps = ldem_2023_max,
        min_mipmaps = ldem_2023_min,
        elev_scale_to_m = ldem_2023.elev_scale_to_m)
    stack_8 = Hyp.TerrainStack(
        Hyp.SiteTerrain(site_8; window = (0, 0, site_8.H, site_8.W)),
        far_2023)

    @testset "1m VIPER 8.0 crop + Barker 2023 20m farfield dimensions" begin
        @test site_8.H == 4144
        @test site_8.W == 5040
        @test site_8.pixel_size_m ≈ 1.0
        @test ldem_2023.H == 30400
        @test ldem_2023.W == 30400
        @test ldem_2023.elev_scale_to_m == 1.0f0
    end

    for (tag, expected) in sort(collect(KNOWN_GOOD_1M_VIPER8_BARKER2023), by = first)
        @testset "1m VIPER 8.0 + Barker 2023 farfield bit-exactness $tag ($(TEST_BACKEND_NAME))" begin
            dt = DateTime(tag, dateformat"yyyy-mm-ddTHH-MM-SS")
            et = Hyp.datetime_to_et(dt)
            sun_t   = Tuple(Hyp.get_body_position(Hyp.NAIF_SUN,   et))
            earth_t = Tuple(Hyp.get_body_position(Hyp.NAIF_EARTH, et))

            sun, dsn, de, sun_rays = Hyp.render_terrain_stack_gpu(
                stack_8, sun_t, earth_t, 0.0;
                backend = TEST_BACKEND,
                DeviceArray = TEST_DEVICE_ARRAY)

            @test sha256_bytes(sun) == expected.sun
            @test sha256_bytes(dsn) == expected.dsn
            @test sha256_bytes(de)  == expected.de
            for k in 1:8
                @test sha256_bytes(view(sun_rays, :, :, k)) ==
                      getfield(expected, Symbol("d_$(k-1)"))
            end

            sun_rgb = _palette_apply(sun, Hyp.SUN_PALETTE)
            dsn_rgb = _palette_apply(dsn, Hyp.DSN_PALETTE)
            @test sha256_bytes(sun_rgb) == expected.sun_rgb
            @test sha256_bytes(dsn_rgb) == expected.dsn_rgb

            ref_sun = _load_png_rgb(joinpath(FIXTURES_1M_UP_TO_DATE, "$(tag)_sun.png"))
            ref_dsn = _load_png_rgb(joinpath(FIXTURES_1M_UP_TO_DATE, "$(tag)_dsn.png"))
            @test ref_sun == sun_rgb
            @test ref_dsn == dsn_rgb
        end
    end
else
    @warn "VIPER 8.0 crop and Barker 2023 LDEM not available — skipping up-to-date 1m bit-exactness tests"
end

end
