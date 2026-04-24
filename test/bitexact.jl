# Cross-platform bit-exactness regression — the canonical verification
# that the pipeline produces identical output across Apple Silicon CPU,
# Apple Metal GPU, and NVIDIA CUDA GPU.
#
# Runs on the KernelAbstractions CPU backend (always available), computes
# SHA-256 of every stage of the pipeline, and compares against hardcoded
# known-good values. Also decodes the 40 committed PNG fixtures in
# test/fixtures/bitexact/ and verifies pixel equality — so a regression
# is detectable both algorithmically (hash diff) and visually (a failing
# timestamp points you at two reference PNGs you can open and compare).
#
# If this test fails, either:
#   (a) you changed the kernel math intentionally → regenerate baselines
#       (`julia --project scripts/bitexact_test.jl` on every backend you
#        support, diff with scripts/diff_bitexact_shas.jl, then copy
#        `data/outputs/bitexact/metal/<ts>/sun.png,dsn.png` to
#        `test/fixtures/bitexact/<ts>_{sun,dsn}.png`, and refresh the
#        KNOWN_GOOD table below), or
#   (b) cross-vendor determinism regressed → see
#       docs/cross-vendor-determinism.md for the 14 documented sources
#       of FP divergence and how to audit.
#
# Skipped if the LDEM is not available (flagged as HAS_LDEM in runtests.jl).
# Runtime on CPU backend: ~10-15 minutes for 20 timestamps at 896×512.

using Dates
using KernelAbstractions: CPU
import FileIO

const FIXTURES = joinpath(@__DIR__, "fixtures", "bitexact")

# Known-good SHAs from the 3-way verified state on julia 1.11.5 +
# KernelAbstractions 0.9.41 + Metal 1.9.3 + CUDA 5.9.0. Every entry
# matched bit-for-bit across Apple CPU ↔ Metal ↔ NVIDIA CUDA.
const KNOWN_GOOD = Dict{String, NamedTuple}(
    "2027-01-05T00-00-00" => (
        sun     = "7caf60fdbfbe931d3eb66fc8d93128737bb3adb46dc6b83c7a63be29077b6161",
        dsn     = "da424b20dedef9a2e25d2e6abc9935cf9dc24c2a085e479976b3ed45794c7904",
        sun_rgb = "2007cc05f7d855f22dfaf5a22ab4bdde4abd66d24b02a8c479b49e0f7a99b4ea",
        dsn_rgb = "d52469ad52a18d051ebc672aae84c93c1967ce70ce14941251bdd376a7863349",
        de      = "6a10b39b3dadce13d7d21a0ec9d0ce1fb9a0a3007e4d1a5d814239365348c0d2",
        azel    = "4a5274c6d349648c5c225eca1d8b90ed18b5dfee5ac4372d927c9d35e1545f5f",
        d_0     = "e44f4e917b5ca2070a5485c17037973979abe8b1e2bfcfef852e9827887d1bbb",
        d_1     = "16a7e0f07608c2223f933cca9fba0b5c0606a375bb79f244ba4fc2a29cf2e5e0",
        d_2     = "ddc7f759921aa59ed778850431db905d8923e33d77b870fed2e664eb4465f849",
        d_3     = "561fcc2817d279a7ef3247139b5c98a1fcd4178e7d8d75bce5109af08da2fb56",
        d_4     = "f13b105360dc9f6f7e106dd5562c48a06655fe3f514508a047ec845094c725a7",
        d_5     = "66bcd3fcebd04c884ad74955ec23bf32c2af8e92622bd5b1b95d62a96622db6d",
        d_6     = "d20d242f20c6e6bfd1a5e8cfb03fab6e6ba58b2d25ab1324ab10191bd417fa4c",
        d_7     = "ae336f592424a5c86f183ca05ed62baceea569cc834389ad68b86935a6b080ca",
    ),
    "2027-01-22T07-00-00" => (
        sun     = "c566934f1563defb0b96afed8c5575277f97dbb78f565562333d4f1beb48278e",
        dsn     = "ce9a2f1188a596a6eb8542b0a77daecfc886b4a267f61e6c822e2b0db7da55cb",
        sun_rgb = "621558e77e1480fd7f167187a3cab7f0385f7aebf5ee3c3910ce1621ebabf0f4",
        dsn_rgb = "bf3f1cbe1905f1afb745f18950a29c69d57e0663844b382a3e44bd9c57f7049a",
        de      = "72dbb0f1f0ab76057d0ea160aa969f29c130bfac7665b445cbd24d5fb31e090a",
        azel    = "14dd62b80ef8f82307c1b5b1ed1dff1d1cad1def316b570ef025e7b8ff589666",
        d_0     = "a61d72ef3079525644b24807d2e6b5aca4a4458de48ef3a1113e80cc94bc79c8",
        d_1     = "e231d7fe5df5c56556e4fdfe1e176037f3e9e2ba75c8a546f004e797fec66aa7",
        d_2     = "cbde014022b1572d7e830cfeaf503aeed2d71d60943b47650ca17d2afd443c56",
        d_3     = "18f53fd3643e155d195781c4b11487f0881452003e1cea21b599f370c2d24f1b",
        d_4     = "757364e7368030c0fb8f48ed2187793bc553794bf19427924e6cf16d5ac9684d",
        d_5     = "fcb1bd2ff1c537e8be99223f3023dd1b63e6dc3a844b3ef511f01a8cf482f930",
        d_6     = "cbb589a72705766854103733ba69d028e87e0346c07ddee83ed6d05fad47484b",
        d_7     = "5b2ff08b5344721d3d688716d47aff8da3ee5c7824c4f263bbc93b59a0e08dfc",
    ),
    "2027-02-12T12-00-00" => (
        sun     = "04dc482d4efdcff084ed62f9845e4b69ebd627de3a7ada44596bd33abd71c96f",
        dsn     = "f48b68d6273866ea8c406864907759cf2be56eec02aa139a0b4301a20dc1ae89",
        sun_rgb = "94e9bf301b36e6b45b0c3138bfc960e9a9d05d892443bafde734edcdb100dbdc",
        dsn_rgb = "a19de68dcd18cb3157137f459c47516c6f5f9ba0eb1544ac98e5e0b119e2303e",
        de      = "de5b3c2b9cc56b5609fbf95c01d9f3e4c0a7cf21ad612adb229bcb0fab86004f",
        azel    = "8311ea152764fce0873b612c9f00d6157dbe283910eeae7de409d188e0ab20c4",
        d_0     = "f0f6b7d40bc0d4f49171461a977a967f50c3069a280db4db2bc53890a3639fc7",
        d_1     = "60479326ab31a3321f75a71d7a6642b74690e7e1f81b1a6715d3df97614e95d5",
        d_2     = "38efcae7082318481c49bf2169a77a6965ca5ffdff1da6cc35e7609c8296fce0",
        d_3     = "610fe947e1b7cf16df03a834897b2982c13f7d8b254707e57c9b602ac92b5424",
        d_4     = "6f002fa2c0f10b5948f686021a64ea129ea864bd72a6da81a1c1c76358495e4e",
        d_5     = "475a43f3027521816c6f9b7a2721b36acae110bd9509a6c2f7d42d30a223cdf2",
        d_6     = "32f9dc34a0effcc8de28030885367b343ec5676dce29fda0c93d751e057604c6",
        d_7     = "a4a141f8553de86696437f6b56047b008ceff0612f1c345f8a46bf8a57bdc1ba",
    ),
    "2027-02-28T04-00-00" => (
        sun     = "9d745456f6ce0efb323b1ceb98484b6b63856118837fd20ca0df64bde645406b",
        dsn     = "63984f9dc64bb2cf75614f9a992bc9974b4f1fb7039bd6877ca1daee7cb8f8e2",
        sun_rgb = "4c7ef5889dd5c835e717f4ddf33f90e82c665af94272aa5be3c7008a92a35a80",
        dsn_rgb = "af45317d7420025d1e0bceaef71ea5b68bc6e6a8595a13b3f1a45b0cd84d8f5b",
        de      = "8c24b807c98a22ece5705d4926636372cf59f730790d5df045b8fb9428384c73",
        azel    = "7e6ecfc47cb8186e6e02c322febd179c7659dd2e6e14489db3c0b81a02e85469",
        d_0     = "416d5711b9ee150edfc34b2bc9d800005f36dfb2004b953e5bc8562aef24386b",
        d_1     = "6dff76a3d14192c95894ce986cf97d86673f52a47d88d65ec5ff0cb45e89e5c2",
        d_2     = "2a51ef2b685e69e6c6a05383a3b6edd607600cde1aaf1cee6dd72e7afb093fc8",
        d_3     = "efc25fc8de414b71b0fbcec61fc3d247b9f05fa95f17addf7f2369fd8c6170f7",
        d_4     = "0c40ce9355ca9fc2218e1499596a90653163f320463edb32c7be8538dbbf35ca",
        d_5     = "76a23cfcaab99fe7c7175a26472a5fb0c3368b2b0e1b4ac386cde1c71dd549f9",
        d_6     = "eb1faed3f5eab748d3b44fb17fcd15feae947b9196b180aaf5cfc4a7f8033ed8",
        d_7     = "5c3c5fc9975c00fa146560757e4b0221a5faf790e125b43eddfc753fc30a3cf8",
    ),
    "2027-03-20T06-00-00" => (
        sun     = "cd880a2495ee64f029058f619432a530d3f05874f73633f91d93aee379eb70da",
        dsn     = "7cdba5df7dcf276d9c0b361e2474084b7483399e2f319732cc4a7be5b5d53d42",
        sun_rgb = "b7a14255572795649e82ff0f518251db45195ff830df8d40b3df00ad5545828e",
        dsn_rgb = "bfe2b9388fbc1535c6565b65049fee7fc7eb26d1c7dc1e60d7c9cd419f2a42f1",
        de      = "d0161e43878dba8e48c9db2cfad21572a863b5bab7cb4a8c3718a3b0829cb04c",
        azel    = "fe4bc313fe7875e8b3fc845798bd55c4efd017593e407caab2aba33234882c3e",
        d_0     = "9dbd10913904502d1303b45ed77c7cae47c0afeceacafdee028225305ae2e2d4",
        d_1     = "fcb0b3bfa88812674ec71cd7751f3a0e8d9bab5ccd125adef8fa731b03a7f55f",
        d_2     = "5edd14b8708d7599115282f1a97d007e1116e09d28421ccb56842ef11e67df0f",
        d_3     = "0d1cdeec621b2a790ff5878b0d1838defaa0a79ede5134473d00e21941f4d0fb",
        d_4     = "dad20bac8bbeff6386bee9b28d2c79e340f1ce97774a84d382b13fcc6a43c179",
        d_5     = "0cdbc95dae7e7bda13dc42ef973df0956a073221e53c7882fea3c7f966600e66",
        d_6     = "7c5a671051fe7e25c1b9646139aea9bc7367a9ff2b9d83fcd86c21140f6f9438",
        d_7     = "81b1d21c8bf66c24e54727da650f2fd073523fba0af4f3d6a034e73ffdaf2763",
    ),
    "2027-04-08T18-00-00" => (
        sun     = "bd91de530e10057e41b1bcfa1b00d1fe24eb87bf533bcd28251a51a59b2ff085",
        dsn     = "7d736cf2cab91d7390790d75aff62f7d8a8a3863bdf7205a5a1e59cda3b05460",
        sun_rgb = "a184b67fdf96ad38055de40738136749dc637f3d68e92050995b8aecf90111b2",
        dsn_rgb = "a624faacd4aa35bc2b0c09433dd0b175c30959fef6114c0645f90719983dd013",
        de      = "70c66a68550106b8ac467f60fe42b1877b98ce52640779f3968a102b2d498ba2",
        azel    = "4ee715efd4f7f4a5bcdc01e04e68896399126a9b6cff1a24ad8e46ca65746649",
        d_0     = "b9e1a92799b57a74997d9d7d8553f99ff63fdb79c34bc52ca98c3098c82f287e",
        d_1     = "100de3c011f42d323f3662f15ced9aec3b1201b8df810ff8c1d48c5adb42f6b6",
        d_2     = "8fde49444ef1caaaf6ea3ee67f83e22376578544a4865f5b224415890ca897c6",
        d_3     = "05a4c040af99f119495914fd861e4ee3cee8f09c1a1d591ca3d56a8f6071a911",
        d_4     = "8b981cea6f0bc5e3613bc8eb959600b971865b1ae726158705a36b4a407bad80",
        d_5     = "6d5c08a2b16505969baad319e2f448b1bf312fc5faf298e34f183ba68a76e8dc",
        d_6     = "d9cc9f119a866896f703ebd6bc99ee261f72db9276aebea3673c6769c4f90953",
        d_7     = "8a603b188d28c70c7d3808daad0baa740510276c4227f5b2d92dfe28df0a9eb4",
    ),
    "2027-05-15T00-00-00" => (
        sun     = "0c3dff88b0db8849626440e87135699aba4f63ded68ae409013584b9f1d18a99",
        dsn     = "0b0969fa6b2d6a73d269d8d5d155f32387719f449d9b6051eba3eee6e5340516",
        sun_rgb = "35198229f69f89e2c54f57884ebc4471d92414c23afea807e47952f2feccdfeb",
        dsn_rgb = "3eb4f23259fdfee2f43a84988d2b7f097a72c97523c23f19604b188a6a73ba8e",
        de      = "6f18787820b1401dd4bf92034382a2483e9036e3e34b616832c74af25c54bf10",
        azel    = "d451558285bdcba6384b71432f890bdd3b12ce7281f24b9d9055937266325af4",
        d_0     = "6ff9c2ae988ba44955fdf6e202949ca2b68be11d2014cab326f617e05543d3ba",
        d_1     = "1a213556dabc7babb4e332cb183b80af41bc50cbb62b28b2d738d630449755dd",
        d_2     = "b0c384be8bb773247d9e90f310cffe115451b51f7d2e03d53aa73348563ab77b",
        d_3     = "850190f82e39a0e15b50dccbe8fa7fad84dc04c06b1493c95c0e8481ba4bdbff",
        d_4     = "9fa99b48f5c1ceb0adc41a6f22b6362c2eaa818e6641b1dbd2107a63c2ea473e",
        d_5     = "9663b8bcf96bbb9e313c5887125c190dd5e88a722ff4f3161b14fc48544d0a56",
        d_6     = "66f490afc412a8b5ad54e05ab89afb34f1ef0f2f474f9c4ad4988c4e04f3ac93",
        d_7     = "f5635443c9d2768f4fdf7b22ce515b72806acad8a4d01aa01225170bd9e06de2",
    ),
    "2027-05-24T19-00-00" => (
        sun     = "6d434f67ab7f133a8dacca353be7e5886c7fe6ed958432bf656f9c6e04ac6d96",
        dsn     = "777113855406bced2862e01b738f197bbc1b37a8605d767f46987dc3aea5ec5c",
        sun_rgb = "593690c54722069cb213c0e925176e40a4aafc95d40ff1c812dc24caedf4ea4e",
        dsn_rgb = "33f1da67505f946a28a44392bd9317c0ab9393618a03a602f6a79e8194f469c9",
        de      = "b192f00732af907684b4e848e19f741a2ad520f64671ec7c8564def8c2802019",
        azel    = "0eb19dce2dafd99fef7454efe14797ae8af09d120cdbdbb6036c7f951c560f49",
        d_0     = "cff58b2ca77e1b7a6916982eba2fef8592bd9932848eee35cc9c334ac31ba018",
        d_1     = "52d6c609c3c5d8a8deeb8bb3e5baeaa76fddf7f0f0fdc9532f5cad09e981defa",
        d_2     = "17877aecb746f51efa38b0e63a4791763e971d07b2a68dac9d1cc64aa0745308",
        d_3     = "c35182f954738e8f34e8bdff8b88227e99ea3f31eb8897a1c1c1100caab6363c",
        d_4     = "10f2d1931d9d15088da159231d8df0c4c5dd8928ece0c630b76818e694135ae0",
        d_5     = "852d1892f5cd2b7c78374a9544424bdd23eabb58aa304b62171b6700ccabb2cc",
        d_6     = "e656b9854ec62389dccfcd71d22cf2f7defb92985b24412c0df4386aa4bce7d4",
        d_7     = "fdb0eeff1a8bf969781c427b58a907f047c0432c0399f573290d63a091763483",
    ),
    "2027-06-01T00-00-00" => (
        sun     = "183facbc37f3a18e7fbdd26d4963c44885692139b16d1da7a41da17e5770f9e1",
        dsn     = "026939d8ca8ea084a302268844921b3f9b533e8691dcd7b344ec70f3e7f522b9",
        sun_rgb = "9794838a8913f06045b1b48909d103d4208f6a8050845b796cdcc20e5600452f",
        dsn_rgb = "805893a70dda8eee9ebf1520e86b8da1b9f435f931fb6b5679d0d9a1f619f67c",
        de      = "a3dc3bf647e55ddd0d3b1e6609634e3a8e3c6af0a11e8a006311ef98fc8431e9",
        azel    = "74991825f0839f7b115d3c3c693c360414a01e81308de4cd6830787d43fa4366",
        d_0     = "ae6d2e01666acd74eefc0cebafe1c4b0ea5f8cd077b7e8db8a94719903a60455",
        d_1     = "1303ee9038a3ab0c3e4a04607abca1af1db674f842a4102b1ab2417ad240bb14",
        d_2     = "8618ef16929c5575593c2c2d56ea514dcb38799e27ee684c72f6739a585ca8e3",
        d_3     = "1b657a47457e3ac150f145051c8c5b2217a82198f9194831ffc9b21288280e7f",
        d_4     = "eae7c5eda601d447b2f4dfbb04c2424a00d043ca0f1658ca1e42dc8c0677f0d8",
        d_5     = "0183c8bc1a1da0fc3c2348be94d5b66f6e1c12b41d5bf7574838e93d769ccc7b",
        d_6     = "894f556720cce09a326cd6e97d5251f15934773d138499730ae9569bb4413b81",
        d_7     = "a162ee7f5f252f3bdbc65e8484b3bc7e4c1a62f14e6f4aff7102596b3b323db0",
    ),
    "2027-06-21T12-00-00" => (
        sun     = "de5d650c7c6f35c50a0a4c356c2d91fe04152ba85f35eb490dc429bc84a1ac13",
        dsn     = "e84375d298b345572d4b6474b1e403bd857dcd9fad8552de08064d3cd227bd4f",
        sun_rgb = "b4640966b649a6041f1a621a3abc44ebed64115a4f0d56c86d96e391e5b848fd",
        dsn_rgb = "fb39a6159fe37b336cca63079fbc7c788007c5c9d6850e8fe5df64fda3a4c63b",
        de      = "57bcf60eca5b11000dc1d5f509687e852315736a63166a3b376e3f920cd42a53",
        azel    = "c71d434e94fbac0fac12dd549d76c52c80daeb59890d6715b6c70a91ba1bf3fb",
        d_0     = "7a39596c1ab47f861e9348b0c9339a7c1619c9722662f49b416bae3c82433272",
        d_1     = "8c38c56bf4d7e44df85e59ac97b39e1b0161736e8feacb3f806b820ec0874f36",
        d_2     = "caada158a58d926f3df56b19337d6a2d108082009dcb5533805bde58b7c0c677",
        d_3     = "6659b75ac0b752fab5f81c01735ecaf965a52c2fff841b317aa45abce4fff4ee",
        d_4     = "aa9c3554495693ecd94cafa63d26d3d1acd67ec0231d9ea9fe35f3fff0603324",
        d_5     = "c30be89cc17735b759647556357a8bfa480268be6efb6b7088239bfbbf346d4f",
        d_6     = "ddfc321421d989fb9f3508e6a7da330a7d9db4267b82a82c195a00b063518d0d",
        d_7     = "ebcf28e8d19f25d36dc768708fad2d9ec492087e4640d5cbf7130f3a5229bfdf",
    ),
    "2027-06-23T00-00-00" => (
        sun     = "398181face86433e9f825123cbd614473fbd1540937568342e62e7bcfd3d5962",
        dsn     = "8f409f8489b0c88d23034da4ec9b2089945de714636e7f6b7edefe0a496e7a25",
        sun_rgb = "167decfba8573614767a438af883cc44402adf56b648cfa074cd1fe3a1aeedca",
        dsn_rgb = "cd62d26e683b858060e77a94f62658bcabe2e4c5fa089c4d6d6622fbebad6498",
        de      = "575599dc75ba6aa475ed21aaa322a3cc2d5796a3f7e1f52698ce9c858d9f9ddc",
        azel    = "875ed793e2712b55d75626ddc42eef42c7848da214cb7a3732ae734df64b4933",
        d_0     = "ae70b0d2dde505a9810c3f4c832250d689a8c9a94e457c2b0d12f8de1adf3ff6",
        d_1     = "ef406515539a9bb10897236ebc3cebd6f37c8ee461dcc5f363af1ecbade274d5",
        d_2     = "3e219aa0ab0c6c217b8ea2ea0f09c7d5b0c4183cebe32af40f11920fec205bfc",
        d_3     = "50abc5dee48293afb176c4d75b2f8fb26c09d7e66cb5da3801930cf3730163d8",
        d_4     = "fa9c355d752234f4b9e50fc9a2ef7d66ed0f17289688c16a655ce45c429ce154",
        d_5     = "ab499dadf7ca61c10b5678fdac911050058035237817794f47123a72ec0810ce",
        d_6     = "d9377b3e2714a7b6abfac33779f91013310bc42fd851d1b3f926ab96ddf2adfc",
        d_7     = "951c96a11e9b56f5967704d9cf931c8c68f5b14ab4e7983da625907f011b38e5",
    ),
    "2027-07-04T03-00-00" => (
        sun     = "882cb8cf8e6d4b87711b33e21b5dbce294d300a83fc8aaf77d3c5dcc0c80ec04",
        dsn     = "08b52546b30311b7f53294f8172f2b1049540e23ca70dd8d57a08230b435d1a4",
        sun_rgb = "14769c35be6d4a84a0b26cbd18b5ecc1466942f031b91c3acaccda5d392a7c30",
        dsn_rgb = "14ecb2b8c6d56723ba7ccd2d8a976b6e6262f16fc4c6a4bcd61e1ba9080a856e",
        de      = "3defd03588490b682091575058e751d7e9e6fa76756f2e9f5189fa7647004b98",
        azel    = "da0f32c07cfcf2e733763a4e1b2fffef4106b8c82988fae02e87c9a46e0a1616",
        d_0     = "49038415f69474c8cb8a88abc77e90bf4583ce34e0a62da0b2e09be32619d0e3",
        d_1     = "669684c290607e9560718f35ad6637aed083b4f747d5ab75751e97c55a28bb1c",
        d_2     = "c0cbac36321435f8a3f88e62c568e043d0776e79e9d63248c01472b3d9b28045",
        d_3     = "64cc54403d097e09b50d30d8df8430b8e9b4262b8a7c26ea98298057d7616686",
        d_4     = "5a852dcb24c833b85e71cd6f966aa51c520eff8d3775a58eceb4d3c277087801",
        d_5     = "a0d50b32eb5d81305c4ede06c5873623685f416dd8252f662a049574d0dd7674",
        d_6     = "cef29c0615094986bbdbc8e476a4619e9d7c2a0e60bd6e1fc512ebd3b87c41ff",
        d_7     = "22ba67c283d87dccb5f08a72f4f80dfd70bf1e4c0d7ac5c5a9447d61d802d0c7",
    ),
    "2027-07-16T08-00-00" => (
        sun     = "b9eb98fb5ce09fc01424f4dca0b5cae3016bc5b3545e84a09a16fd22dcfbfa58",
        dsn     = "a4ba97d39638d78f3f9827916602b948fa22757ff535035bc797c78bc1f388b9",
        sun_rgb = "a3c29ccd4d709faa268ea0e1e7f527e8a44a26f57cd8c8f74056c0b38d12dd9f",
        dsn_rgb = "c0e054fcd014d111a47615bae9c53cbdb14f8529bc64ce44e4ba5bd416ec949a",
        de      = "0c7c3cd66330c389c2edcbdf9043be3ea44f6988987d78bd45c7e5d10164e5ee",
        azel    = "8f43e54452cffbf87282f81825fa36509d793a981e9579906696cbe0773507e2",
        d_0     = "3f41e02e9706fea3a2fd68f44ff97e76198b94248e9d2eb53c696e4a8a8b8496",
        d_1     = "64c48ca7f7ad60e7c91764ef28d9c4a7c09d4c52dd697233e953c8f1078afbc5",
        d_2     = "67712544a0f52feb40543450b5be7236d607e093c8da68a01a8fb4962f74bf0a",
        d_3     = "90e0d4f1f0fbd30179cd921900b1705352e9dc5267d3a747e3dc75b26b29108d",
        d_4     = "d27c8488a1cd4ece4cd63812c2d7001f8523ec5c2baef840de9b88f754ac71d5",
        d_5     = "4be834d82979cbe33f11815d758cc37fe7828c6e47424c3ad1e2da6eb92832b1",
        d_6     = "94d3846a3b0369464571bba264b17004b31b2a5dee6484168e3901be2f77e925",
        d_7     = "dd5dc89ba241ebef0fa60fe325d8950a48f1b9dcb29e0225ed2732fd4dab858a",
    ),
    "2027-08-05T11-00-00" => (
        sun     = "a06d618f43395787edb6bde8ef519e76bcfe8fd683e4b618199e10adb4eda3a3",
        dsn     = "806b46e52b0ce328121ded9838a5f5e2d06d21e2bc327b1861dd46b3d80bc176",
        sun_rgb = "5008e6c9711f118378f6264a4400bdfb19024c7b1c0cac930ccafe66c1046f5a",
        dsn_rgb = "d5d738a19353713557a23895bb88d499ee3c220df31927b61572c59ac668fd63",
        de      = "1bebdc884a40a3692f0f6c90106b2a7bd9acf9d1e1f2378bb2db2b65a510d7bc",
        azel    = "b2b62d632f0df5856bf570effdee045b928c1b1e8f3c4d4eb41ab954e8a249e0",
        d_0     = "28acaade388ba79251d7548caec7f98fea8781f4ebab7d3ce064a30b3741be3d",
        d_1     = "9c1926c3cfebd1e2991c68b11e00c66327412516212a9f7e8ba6db3c8ff5531c",
        d_2     = "7f712d8508f8e12a6460be640fe5f8f9a0dbb9e6fa35e3c1e59143ebfd7f154b",
        d_3     = "ae32a23cc867aa62cd65212ead87a50c024080ff62184aca935fb5d696144f21",
        d_4     = "d292f5b92c103c9bc55bf9fe3c258b6ed2e9fb80fb5133f924223203ad2dd56e",
        d_5     = "ba703c57d91e9ef0666a2fc742c4d8ad5ef2b735d4d0f6a29cddc24a8f20883d",
        d_6     = "f04989ab5dd7c910faa96e99dcd5506a7ecaba48f4be91a1baec82c0a419299a",
        d_7     = "eca2cd39cd45a287514f92eca4acbffab68b2b96d50bc8455a75f4d26bcb3051",
    ),
    "2027-08-20T15-00-00" => (
        sun     = "4d216a644dce850b2e47d7516cd2c823dc919a4aa2995c5736db9a71391ca03a",
        dsn     = "54300b3fea96b077ba3252b0608621091c3c3522c1cffba0c140f2fc008da66d",
        sun_rgb = "88a9567bebab2c617bea013698911bfced548f795a463dff5f95a0f8ab4b8b43",
        dsn_rgb = "f521cdd6a74890e96d8131c5bcddc41e03db1464085594b4b91a3e3e5c9a6767",
        de      = "3f08a9d55a3f93fbccb0d023f98113a787feb41651565875a337c9c8e5d2e201",
        azel    = "53c39179259c434ed87584df805e4fa94d1bb5240a0c0d9f34dd75629ee11878",
        d_0     = "8cb12c6f549f70bc8152be9ddd50ecc67dbf492a4a629559322ebddbdca4d4f3",
        d_1     = "aa4d1e11c092653954a6a24c2ac74e9b1cdf125033822eeb8f0ea598ccf438c8",
        d_2     = "29c7970da845569ca5a96bd5fceadeccd4ba535196da4d2a3fa46a0a3c9e6754",
        d_3     = "98dce2a38ff201f56b25759f765ea6707dc812c996e601ac40814220d4dffef9",
        d_4     = "878533b6f340ba424331658521f4bbcdb64227715a0ca5b7d38c9a03592eeae2",
        d_5     = "66af846624000461af784adfc1535d96ac4fdc2c397f4da3743007a20d24e06f",
        d_6     = "286629ff18c6e55f2468826f6732f5fc4678de084737b48d8badcd71dc081298",
        d_7     = "0907d6508d3629a036ddb25e9dd9bc5550314a38c5a0379e522f7c4136315fe0",
    ),
    "2027-09-10T00-00-00" => (
        sun     = "9ac6232327d430d28b21b797e8204237eb91ef73cb99bcce2f8a018a081e2cd8",
        dsn     = "805228f8bc930fcfb61f369940b909ed4decbfdc341148d4a96181cb053d6255",
        sun_rgb = "a46f58b14e993cc6d9523fb03e9821e2a3729faec32df4a2ad1a6e6e490ad3e0",
        dsn_rgb = "c1c614a4303d01dc16bf0eca2f0976c0728f487f13ead9d2d908becf6bf55458",
        de      = "8222a80f9cf4fe7c9e749243acb9fe4e7b770a2e432ac5369dcdd384dc3abbf4",
        azel    = "f83b864e6605aa9bb52dfadc67e9345fcf749608f0628bb7b8f0aaac1a9f8bea",
        d_0     = "289158f079a779329194590d73effb7a22c55c26c3e943dc8b1c27d9ab1af1eb",
        d_1     = "1e901f088379c3e2f65a25772b6860336cc79b2951c0776451757742a81d001b",
        d_2     = "7db9b74e914225a6223a93d17407844d8d5f070e7d84fd91828fd57d80420b68",
        d_3     = "6ce2fffa076905a4a13d697d497460bb16d23fc7f375de6f01613f7c3fe83a63",
        d_4     = "1752d7dc2581a6913e87f58b2e6a9c5f2505ef58a0e24d96e28b3b9933b74a9a",
        d_5     = "42e959801fc706f5368412d21aca4d20272b136669f9c479dd46316829003ec5",
        d_6     = "9268423b0011bc90db5cc76ec2781f276960862eb09ed9ef52e92cf394e30f00",
        d_7     = "cd4d2bc07bb0e6435fc1e0ee46a5f74d720878185c4bd256fec03b4fe83c3dcc",
    ),
    "2027-10-05T21-00-00" => (
        sun     = "9bfb47a94cc9d7d913e405c720856086ff7a240edb8184614b278f0333c20c0d",
        dsn     = "50e2e8d88b6ab4b7f79e4f345f9509cfff321394976f8ea53ce7da405edc5b65",
        sun_rgb = "ae5efa66c25d611e682ed5fe2d32610951a8312cf012190df3fdf07407999e41",
        dsn_rgb = "bd6c0a86a0fb479f4e766f2cc5c0e90ab26e63fd72f72c824febbc1b1e15d88b",
        de      = "d9e14c300c9e328ebc92d006b09eed5535ec53f96701d5286f0352492bd4b326",
        azel    = "1fc0736570b9c1eb067c6aba2a2aeecee8f1377939b83e30b82720acc078c9b2",
        d_0     = "392e6eb6f872a2be8b0e25ab8b630da2c1950b69a1f9611498cdb920852c5cfc",
        d_1     = "b31d953d29d9dccefdd6ddd7d8ee85f57dec85b3953fda9d4b9cf99e1980f71e",
        d_2     = "0d7a25e2a4ee4777d9813562827ea23567182272c2c27065c9e05ca06cd9eed5",
        d_3     = "41bc2faa9ba77a9332c3db3940cd26565fd449992f66beaedea3754af8e742d1",
        d_4     = "2416ff4afe6fef8aa6f69e624fafd8cb75070e6ab48da3fd5d3f3fef17327cc3",
        d_5     = "31b99832bb4ba22af67386c8cac6bb660e0f039cbb792e69242d0ce7c8cf1b09",
        d_6     = "379885cbc499e8ad3e3efe1671c81db38b6dab227a01799b2887f1d1540ca61d",
        d_7     = "333413ddd3499c5945c83a89f798e7084642a8393c2101e298a267c3b91817f5",
    ),
    "2027-10-27T14-00-00" => (
        sun     = "d74446327be567cb1be54ee31cf9b8eaf9a206a2aabb4e50999650af650691ab",
        dsn     = "4d3d26096ff3a18b54aa1cdba17ca87b8bd8d864823ce4060e60518df7f397ec",
        sun_rgb = "0e04eec3586f69d105e12da70381a2423e22e32513d3ec3148bc93b5b6d6d73b",
        dsn_rgb = "8da3cd9b4cfd433c75fe425ed9817e5022d4fd74a19a21985ee317712857bda9",
        de      = "a56bead119dd0fc07b0ac7cc88c8e9ece5c8eb9189e6dc996b055ea8b7a3db17",
        azel    = "3a0e0382cb9b37d919a1aab33a792c631a625b84cfe402aa2b2b80b0c6c647be",
        d_0     = "5c748d87398f3d9acdce44b9e1a5725f9e8999898b6ed0388138dfad574c2bfa",
        d_1     = "28656d15e2838b2a64557cca9c6c10ea81e2a6357b0e835b17b62d4a553cf7fb",
        d_2     = "d5e6ece85ed9cdb733e71633412b0bfa7e8a0321c126b7c12af705255c20afdc",
        d_3     = "c8b5959d2492d3ad01be8e497717c14b6008cf0866deb9a5e5cabec0b8e1d703",
        d_4     = "42298ee73d0e119b036ee2f68588d12aadd5777cbdbb1f0abd838fa12c33cf2b",
        d_5     = "4f20a358d89bcf58239dc3539f9a75ee60e920a38b59fb67127023594cd3fc8c",
        d_6     = "888e428e43b2ffbf4f058a0798f32167bbc1e11aed3b3ba477d06a84b76c5d01",
        d_7     = "411979273440f0ab0c803147d190e97109215cd5f8905334e4451fea7b5785ce",
    ),
    "2027-11-11T12-00-00" => (
        sun     = "575279f35057d05ba7eed47eb524574064aaf453a688faef14dce4a948f147f0",
        dsn     = "d512f71a378921e2d4e2405344cdac643eb0e856e2932bade017ba66cfde9159",
        sun_rgb = "f32c152e6d04a71096b735d7b623b5ac1dee85c04d42bd3b900576e42fe83db6",
        dsn_rgb = "2b766ed38b802ad55eb4e09003ee1dabd3317bedc7985c52c4b85a3365dedf12",
        de      = "93e58a192eee251d23aafddda10c61c0a3b4fb223191c561d6fdfc94f656455e",
        azel    = "73da292d350ad6b4e91f42b30e24888df3ade8c839c8da391688f982e70d98ac",
        d_0     = "dad01ef5ed3d1a9d3c1f7949dd96256026ecad135151d5f87052aee2fb3c4d4b",
        d_1     = "7da3183437521799c9e4c581f295b7eab310b04270638ba0816cf0e2370892b0",
        d_2     = "6ea7163ccea244004421b0f1d4b83ad2ed40afc04d55984b3349f686a442b8b2",
        d_3     = "3363d4e4a2ff093d67c057c0c54beef74e4cb6b5d87374c90bdd9828b115ddfe",
        d_4     = "fd61f6e5a8c6043b78f96e7885576deb5415162eaa6edce85f361a85e1539e2a",
        d_5     = "50d8d21f512b1e733368070aafa0b2cdd150607151c571d088dac4cdd7f0d0ea",
        d_6     = "f79384b69ff588b4da2352970623a7580ea2819e4153f088e3d870a008a5d320",
        d_7     = "e5bbc2ba94f98e134862aa506c329ab3ea2aa654ff8cb858cf95dfeff40c7172",
    ),
    "2027-12-21T00-00-00" => (
        sun     = "8bc9d420643a4697fd11c94a03123b3862e7aa041cd4302240d0ada129a77a50",
        dsn     = "b8c55e1cd994ecfcb71c8766e77fab53392d265339870191a6e7286838db598b",
        sun_rgb = "c887f07c4b7f6b3ec153d558b23d6006f8dbccbe8e069577cb6d272f937067ee",
        dsn_rgb = "7a4f9818546259feb3d51ec9396a0eaaeb36af9651658d3931ae892ad648969f",
        de      = "b4ccdf82b37c6c7242ba398cf0876de3219cad2696b4e40a8711a10b40aeae12",
        azel    = "cbc62238b56e338f4d69513c377bea9cf99a0006d70be4ea4929397583286486",
        d_0     = "859db1ab2356191964012ea7ca62c3d21482d7c0fc5ad02688b7e4d5c7885a93",
        d_1     = "db418306c718e26080e8597bb761005b3be0efa0382eb78bead61ed8f0b7d23c",
        d_2     = "60e665203b1f3295b73c4ddb25a0407ef3258c38694896c3bf741f2f7b9ca283",
        d_3     = "bf926ad420d847d1e3823c568314cbe1c05ec1884df2ce499ca7066624cac967",
        d_4     = "95bc6b38e1f0d3881d08ad1e44dafceee95aa1ac129c737714f0eb35b9406da0",
        d_5     = "87535f8a47d034308d76e773349d56aec854d647e19af1c8d2857b8358dce16f",
        d_6     = "af5de2186b6a74c293b8cf27b084e67ddad98e413569f5e9833b0b0d51a9faa9",
        d_7     = "e4c2ff0ad10c0665e9cdd4a1024eaa50dc4ff82df5bf34d0c5357a5327c5b92b",
    ),
)

# Palette application — must match scripts/bitexact_test.jl's version
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

@testset "Cross-platform bit-exactness (20 timestamps)" begin
    ldem = JM.load_ldem(LDEM_PATH)
    JM.init_spice(joinpath(PROJECT_ROOT, "kernels"))
    max_mm, min_mm = JM.build_ldem_mipmaps_minmax(ldem.data)
    origin_r, origin_c = 8960, 18432
    H, W = 512, 896

    for (tag, expected) in sort(collect(KNOWN_GOOD), by = first)
        @testset "$tag" begin
            dt = DateTime(tag, dateformat"yyyy-mm-ddTHH-MM-SS")
            et = JM.datetime_to_et(dt)
            sun_t   = Tuple(JM.get_body_position(JM.NAIF_SUN,   et))
            earth_t = Tuple(JM.get_body_position(JM.NAIF_EARTH, et))

            # Run kernel on the always-available CPU backend. Output matches
            # Metal and CUDA bit-for-bit per the verified 3-way invariant.
            sun, dsn, de, sun_rays = JM.generate_live_shadow_frame_gpu(
                ldem.data, origin_r, origin_c, H, W, sun_t, earth_t, 0.0;
                max_mipmaps = max_mm, min_mipmaps = min_mm,
                backend = CPU(), DeviceArray = Array)

            # CPU precompute buffer — its SHA is part of the audit trail
            # because a drift in CPU math would silently corrupt GPU input.
            sun_rc, sun_rs, sun_el, earth_rc, earth_rs, earth_el, sun_tan, dsn_tan =
                JM._precompute_azel(ldem.data, origin_r, origin_c, H, W,
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
            sun_rgb = _palette_apply(sun, JM.SUN_PALETTE)
            dsn_rgb = _palette_apply(dsn, JM.DSN_PALETTE)
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
end
