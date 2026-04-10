├── CodeBert
├── LICENSE
├── README.md
├── all_in_one.png
├── bsc_ast_12170_12190.txt
├── build
├── configs
├── data
│   ├── README.md
│   ├── embeddings
│   │   ├── BSC_500
│   │   ├── Ethereum_500
│   │   ├── Fantom_500
│   │   └── Polygon_500
│   ├── graphs_ast
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_ast_graph
│   │   └── Arbitrum.jsonl
│   ├── graphs_ast_light
│   │   ├── Avalanche.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_ast_llm
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_ast_llm_fixed
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_ast_norm
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_ast_norm_uid
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_ast_norm_uid_v2
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_cfg
│   │   ├── Arbitrum.jsonl
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_cfg_contract
│   │   ├── Arbitrum.jsonl
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_cfg_norm
│   │   ├── Arbitrum.jsonl
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_dfg
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── BSC_500.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   ├── Fantom_500.jsonl
│   │   ├── Polygon.jsonl
│   │   └── Polygon_500.jsonl
│   ├── graphs_dfg_norm
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_multigraph
│   │   ├── Avalanche.jsonl
│   │   └── BSC.jsonl
│   ├── graphs_multigraph_stream
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── graphs_raw
│   │   ├── Avalanche.jsonl
│   │   ├── BSC.jsonl
│   │   ├── Ethereum.jsonl
│   │   ├── Fantom.jsonl
│   │   └── Polygon.jsonl
│   ├── index
│   │   ├── Arbitrum_filename2id.json
│   │   ├── Arbitrum_src2id.json
│   │   ├── Avalanche_filename2id.json
│   │   ├── Avalanche_src2id.json
│   │   ├── BSC_filename2id.json
│   │   ├── BSC_src2id.json
│   │   ├── Ethereum_filename2id.json
│   │   ├── Ethereum_src2id.json
│   │   ├── Fantom_filename2id.json
│   │   ├── Fantom_src2id.json
│   │   ├── Polygon_filename2id.json
│   │   └── Polygon_src2id.json
│   ├── meta
│   │   └── type_vocab_lite.json
│   ├── raw
│   │   ├── Arbitrum
│   │   ├── Avalanche
│   │   ├── BSC
│   │   ├── Ethereum
│   │   ├── Fantom
│   │   ├── Polygon
│   │   └── Unknown
│   └── train
│       ├── crossgraphnet_lite
│       └── crossgraphnet_lite_labeled
├── docs
├── environment.yml
├── eth10k_resume.log
├── eth_ast_example.json
├── fan_ast_example.json
├── fl.log
├── graph_quality_reports
│   ├── ast_hist.png
│   ├── cfg_hist.png
│   └── dfg_hist.png
├── logs
│   ├── fl
│   │   ├── _summary
│   │   ├── fedavg_codebert_frozen_r20_4c_llm.jsonl
│   │   ├── fedavg_codebert_frozen_r20_llm.jsonl
│   │   ├── fedavg_codebert_frozen_r50_4c_llm.jsonl
│   │   ├── fedavg_codebert_frozen_sanity_llm.jsonl
│   │   ├── fedavg_llm_C2_EthBSC_fedavg_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_llm_C2_EthBSC_fedavg_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_llm_C2_EthBSC_fedavg_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_llm_C2_EthFantom_fedavg_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_llm_C2_EthFantom_fedavg_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_llm_C2_EthFantom_fedavg_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_llm_C2_EthPolygon_fedavg_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_llm_C2_EthPolygon_fedavg_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_llm_C2_EthPolygon_fedavg_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_llm_C2_EthereumBSC_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_llm_C2_EthereumBSC_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_llm_C2_EthereumBSC_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_llm_SENS_E_C2_EthereumBSC_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_llm_SENS_E_C2_EthereumBSC_llm_N500_E2_R10_seed42.jsonl
│   │   ├── fedavg_llm_SENS_E_C2_EthereumBSC_llm_N500_E5_R10_seed42.jsonl
│   │   ├── fedavg_llm_SENS_N_C2_EthereumBSC_llm_N1000_E1_R10_seed42.jsonl
│   │   ├── fedavg_llm_SENS_N_C2_EthereumBSC_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_none_C2_EthBSC_fedavg_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_none_C2_EthBSC_fedavg_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_none_C2_EthBSC_fedavg_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_none_C2_EthFantom_fedavg_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_none_C2_EthFantom_fedavg_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_none_C2_EthFantom_fedavg_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_none_C2_EthPolygon_fedavg_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_none_C2_EthPolygon_fedavg_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_none_C2_EthPolygon_fedavg_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_none_C2_EthereumBSC_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_none_C2_EthereumBSC_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_none_C2_EthereumBSC_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_stats_C2_EthBSC_fedavg_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_stats_C2_EthBSC_fedavg_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_stats_C2_EthBSC_fedavg_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_stats_C2_EthFantom_fedavg_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_stats_C2_EthFantom_fedavg_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_stats_C2_EthFantom_fedavg_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_stats_C2_EthPolygon_fedavg_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_stats_C2_EthPolygon_fedavg_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_stats_C2_EthPolygon_fedavg_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_stats_C2_EthereumBSC_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedavg_stats_C2_EthereumBSC_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedavg_stats_C2_EthereumBSC_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedavg_stats_MUscan_EthBSC_stats_mu0_N500_E1_R20_seed42.jsonl
│   │   ├── fedavg_stats_MUscan_EthFantom_stats_mu0_N500_E1_R20_seed42.jsonl
│   │   ├── fedavg_stats_MUscan_EthPolygon_stats_mu0_N500_E1_R20_seed42.jsonl
│   │   ├── fedavg_stats_r20_4c_stats.jsonl
│   │   ├── fedavg_stats_r20_stats.jsonl
│   │   ├── fedavg_stats_r50_4c_stats.jsonl
│   │   ├── fedavg_stats_sanity_stats.jsonl
│   │   ├── fedprox_codebert_frozen_mu0005_4c_llm.jsonl
│   │   ├── fedprox_codebert_frozen_mu005_4c_llm.jsonl
│   │   ├── fedprox_codebert_frozen_r50_4c_llm.jsonl
│   │   ├── fedprox_llm_C2_EthBSC_fedprox_mu0.001_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_llm_C2_EthBSC_fedprox_mu0.001_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_llm_C2_EthBSC_fedprox_mu0.001_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_llm_C2_EthFantom_fedprox_mu0.001_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_llm_C2_EthFantom_fedprox_mu0.001_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_llm_C2_EthFantom_fedprox_mu0.001_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_llm_C2_EthPolygon_fedprox_mu0.001_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_llm_C2_EthPolygon_fedprox_mu0.001_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_llm_C2_EthPolygon_fedprox_mu0.001_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_llm_C2_EthereumBSC_FedProx_mu0.001_llm_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_llm_C2_EthereumBSC_FedProx_mu0.001_llm_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_llm_C2_EthereumBSC_FedProx_mu0.001_llm_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_none_C2_EthBSC_fedprox_mu0.001_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_none_C2_EthBSC_fedprox_mu0.001_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_none_C2_EthBSC_fedprox_mu0.001_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_none_C2_EthFantom_fedprox_mu0.001_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_none_C2_EthFantom_fedprox_mu0.001_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_none_C2_EthFantom_fedprox_mu0.001_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_none_C2_EthPolygon_fedprox_mu0.001_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_none_C2_EthPolygon_fedprox_mu0.001_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_none_C2_EthPolygon_fedprox_mu0.001_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_none_C2_EthereumBSC_FedProx_mu0.001_none_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_none_C2_EthereumBSC_FedProx_mu0.001_none_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_none_C2_EthereumBSC_FedProx_mu0.001_none_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_stats_C2_EthBSC_fedprox_mu0.001_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_stats_C2_EthBSC_fedprox_mu0.001_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_stats_C2_EthBSC_fedprox_mu0.001_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_stats_C2_EthFantom_fedprox_mu0.001_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_stats_C2_EthFantom_fedprox_mu0.001_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_stats_C2_EthFantom_fedprox_mu0.001_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_stats_C2_EthPolygon_fedprox_mu0.001_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_stats_C2_EthPolygon_fedprox_mu0.001_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_stats_C2_EthPolygon_fedprox_mu0.001_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_stats_C2_EthereumBSC_FedProx_mu0.001_stats_N500_E1_R10_seed1.jsonl
│   │   ├── fedprox_stats_C2_EthereumBSC_FedProx_mu0.001_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_stats_C2_EthereumBSC_FedProx_mu0.001_stats_N500_E1_R10_seed7.jsonl
│   │   ├── fedprox_stats_MUscan_EthBSC_stats_mu1e-2_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthBSC_stats_mu1e-3_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthBSC_stats_mu1e-4_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthFantom_stats_mu1e-2_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthFantom_stats_mu1e-3_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthFantom_stats_mu1e-4_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthPolygon_stats_mu1e-2_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthPolygon_stats_mu1e-3_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_MUscan_EthPolygon_stats_mu1e-4_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_SENS_R_C2_EthereumBSC_FedProx_mu0.001_stats_N500_E1_R10_seed42.jsonl
│   │   ├── fedprox_stats_SENS_R_C2_EthereumBSC_FedProx_mu0.001_stats_N500_E1_R20_seed42.jsonl
│   │   ├── fedprox_stats_SENS_R_C2_EthereumBSC_FedProx_mu0.001_stats_N500_E1_R50_seed42.jsonl
│   │   ├── fedprox_stats_eth2bsc_n500_e1_r10_seed1_proto0.jsonl
│   │   ├── fedprox_stats_eth2bsc_n500_e1_r10_seed1_proto1.jsonl
│   │   ├── fedprox_stats_eth2bsc_n500_e1_r10_seed42_proto0.jsonl
│   │   ├── fedprox_stats_eth2bsc_n500_e1_r10_seed42_proto1.jsonl
│   │   ├── fedprox_stats_eth2bsc_n500_e1_r10_seed7_proto0.jsonl
│   │   ├── fedprox_stats_eth2bsc_n500_e1_r10_seed7_proto1.jsonl
│   │   ├── fedprox_stats_eth2ftm_n500_e1_r10_seed1_proto0.jsonl
│   │   ├── fedprox_stats_eth2ftm_n500_e1_r10_seed1_proto1.jsonl
│   │   ├── fedprox_stats_eth2ftm_n500_e1_r10_seed42_proto0.jsonl
│   │   ├── fedprox_stats_eth2ftm_n500_e1_r10_seed42_proto1.jsonl
│   │   ├── fedprox_stats_eth2ftm_n500_e1_r10_seed7_proto0.jsonl
│   │   ├── fedprox_stats_eth2ftm_n500_e1_r10_seed7_proto1.jsonl
│   │   ├── fedprox_stats_eth2poly_n500_e1_r10_seed1_proto0.jsonl
│   │   ├── fedprox_stats_eth2poly_n500_e1_r10_seed1_proto1.jsonl
│   │   ├── fedprox_stats_eth2poly_n500_e1_r10_seed42_proto0.jsonl
│   │   ├── fedprox_stats_eth2poly_n500_e1_r10_seed42_proto1.jsonl
│   │   ├── fedprox_stats_eth2poly_n500_e1_r10_seed7_proto0.jsonl
│   │   ├── fedprox_stats_eth2poly_n500_e1_r10_seed7_proto1.jsonl
│   │   ├── fedprox_stats_mu0005_4c_stats.jsonl
│   │   ├── fedprox_stats_mu005_4c_stats.jsonl
│   │   ├── fedprox_stats_r50_4c_stats.jsonl
│   │   ├── fedprox_stats_smoke_n200_e2_seed42_proto0.jsonl
│   │   ├── fedprox_stats_smoke_n200_e2_seed42_proto1.jsonl
│   │   ├── fedprox_stats_smoke_seed42_proto0.jsonl
│   │   ├── fedprox_stats_smoke_seed42_proto1.jsonl
│   │   ├── fedprox_stats_smoke_seed42_proto1_fixed.jsonl
│   │   ├── figs
│   │   ├── figs_all
│   │   ├── fl_all_results_fedavg_fedprox.csv
│   │   ├── fl_fedavg_all_results.csv
│   │   ├── fl_fedavg_all_results_full.csv
│   │   ├── logs
│   │   ├── plot_fl_curves.py
│   │   ├── summarize_fl.py
│   │   ├── summarize_fl.sh
│   │   └── summarize_fl_jsonl.py
│   ├── fl_evalplus
│   │   ├── fedprox_stats_eth2poly_evalplus_seed1_proto0.jsonl
│   │   ├── fedprox_stats_eth2poly_evalplus_seed1_proto1.jsonl
│   │   ├── fedprox_stats_eth2poly_evalplus_seed42_proto0.jsonl
│   │   ├── fedprox_stats_eth2poly_evalplus_seed42_proto1.jsonl
│   │   ├── fedprox_stats_eth2poly_evalplus_seed7_proto0.jsonl
│   │   ├── fedprox_stats_eth2poly_evalplus_seed7_proto1.jsonl
│   │   ├── fedprox_stats_eth2poly_n500_e1_r10_seed42_proto0_evalplus.jsonl
│   │   └── fedprox_stats_eth2poly_n500_e1_r10_seed42_proto1_evalplus.jsonl
│   └── fl_summary
├── make_dfg_subset.py
├── model_full_best.pt
├── models
│   └── crossgraphnet_lite.pt
├── node_modules
│   ├── node-addon-api
│   │   ├── LICENSE.md
│   │   ├── README.md
│   │   ├── common.gypi
│   │   ├── except.gypi
│   │   ├── index.js
│   │   ├── napi-inl.deprecated.h
│   │   ├── napi-inl.h
│   │   ├── napi.h
│   │   ├── node_addon_api.gyp
│   │   ├── node_api.gyp
│   │   ├── noexcept.gypi
│   │   ├── nothing.c
│   │   ├── package-support.json
│   │   ├── package.json
│   │   └── tools
│   ├── node-gyp-build
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── SECURITY.md
│   │   ├── bin.js
│   │   ├── build-test.js
│   │   ├── index.js
│   │   ├── node-gyp-build.js
│   │   ├── optional.js
│   │   └── package.json
│   ├── solidity-parser-antlr
│   │   ├── CHANGES.md
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── dist
│   │   ├── index.d.ts
│   │   ├── package.json
│   │   └── tslint.json
│   ├── tree-sitter
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── binding.gyp
│   │   ├── build
│   │   ├── index.js
│   │   ├── node-addon-api
│   │   ├── package.json
│   │   ├── src
│   │   ├── tree-sitter.d.ts
│   │   └── vendor
│   ├── tree-sitter-solidity
│   │   ├── LICENSE
│   │   ├── README.md
│   │   ├── binding.gyp
│   │   ├── bindings
│   │   ├── grammar.js
│   │   ├── package.json
│   │   ├── prebuilds
│   │   ├── queries
│   │   ├── src
│   │   ├── tree-sitter-solidity.wasm
│   │   └── tree-sitter.json
│   └── yarn
│       ├── LICENSE
│       ├── README.md
│       ├── bin
│       ├── lib
│       ├── package.json
│       └── preinstall.js
├── notebooks
├── package-lock.json
├── package.json
├── results
│   ├── ablation_proto
│   │   ├── curve_rounds.csv
│   │   ├── final_r10_by_seed.csv
│   │   └── summary_over_seeds.csv
│   ├── ablation_proto_all
│   │   ├── curve_rounds_all.csv
│   │   ├── delta_proto1_minus_proto0_by_target.csv
│   │   ├── final_r10_by_seed_all.csv
│   │   └── summary_over_seeds_all.csv
│   ├── crossgraphnet_full
│   │   ├── eth_to_BSC_500_seed1
│   │   ├── eth_to_BSC_500_seed42
│   │   ├── eth_to_BSC_500_seed7
│   │   ├── eth_to_Fantom_500_seed1
│   │   ├── eth_to_Fantom_500_seed42
│   │   ├── eth_to_Fantom_500_seed7
│   │   ├── eth_to_Polygon_500_seed1
│   │   ├── eth_to_Polygon_500_seed42
│   │   ├── eth_to_Polygon_500_seed7
│   │   └── summary_full.csv
│   ├── crossgraphnet_lite_matrix
│   │   ├── eth_to_BSC_500_seed1
│   │   ├── eth_to_BSC_500_seed42
│   │   ├── eth_to_BSC_500_seed7
│   │   ├── eth_to_Fantom_500_seed1
│   │   ├── eth_to_Fantom_500_seed42
│   │   ├── eth_to_Fantom_500_seed7
│   │   ├── eth_to_Polygon_500_seed1
│   │   ├── eth_to_Polygon_500_seed42
│   │   ├── eth_to_Polygon_500_seed7
│   │   └── summary_lite.csv
│   ├── dataset_stats.csv
│   ├── experiments
│   │   ├── 12_22.md
│   │   ├── ETH_to_BSC_llm
│   │   ├── ETH_to_BSC_stats
│   │   ├── crosschain_runs
│   │   └── result_all.csv
│   ├── figures
│   │   └── polygon_f1_threshold_proto0_vs_proto1.png
│   ├── polygon_evalplus_by_seed.csv
│   ├── polygon_oracle_by_seed.csv
│   ├── polygon_oracle_summary.csv
│   ├── polygon_proto_delta.csv
│   ├── polygon_proto_main_table.csv
│   └── sota_tools
│       ├── crossgraphnet_recall_500.csv
│       ├── lists
│       ├── mythril
│       ├── oyente
│       ├── run_sota_tools.log
│       ├── slither
│       ├── slither_parsed.csv
│       ├── slither_parsed_vuln.csv
│       ├── slither_summary_by_chain.csv
│       ├── slither_summary_vuln_by_chain.csv
│       ├── slither_tiered.csv
│       ├── slither_tiered_summary.csv
│       ├── smartcheck
│       ├── sota_tools_matrix.csv
│       ├── sota_tools_summary.csv
│       └── sota_tools_table.csv
├── run_all_exps_v3.sh
├── run_full_matrix.sh
├── run_lite_matrix.sh
├── scripts
│   ├── archive
│   │   ├── build_graphs_ast_js.py
│   │   ├── merge_clean_graphs.py
│   │   ├── parse_ast.js
│   │   ├── process_large.py
│   │   ├── restucture_sanctuary.py
│   │   ├── run_data_collection.py
│   │   ├── slang_build_ast.js
│   │   ├── spilt_chains.py
│   │   └── verify_graphs.py
│   ├── build_ast_hist_embeddings_500.py
│   ├── build_ast_norm_uid_from_llm.py
│   ├── build_ast_norm_uid_from_llm_filename.py
│   ├── build_cfg_filename_index.py
│   ├── build_cfg_src_index.py
│   ├── build_codebert_embeddings_500.py
│   ├── build_codebert_embeddings_bsc_500.py
│   ├── build_dfg_from_labeled.py
│   ├── build_embeddings_fantom.py
│   ├── build_embeddings_poly_500.py
│   ├── build_graphs_ast_from_raw.py
│   ├── build_graphs_ast_light.py
│   ├── build_graphs_ast_node.py
│   ├── build_graphs_ast_treesitter.py
│   ├── build_graphs_cfg.py
│   ├── build_graphs_dfg.py
│   ├── build_graphs_multigraph.py
│   ├── build_graphs_slither.py
│   ├── build_labels_with_slither.py
│   ├── build_llm_embeddings_500.py
│   ├── build_multigraph_dataset_stream.py
│   ├── build_polygon_500.py
│   ├── build_tfidf_embeddings_500.py
│   ├── build_train_jsonl_crossgraphnet_lite.py
│   ├── build_type_vocab.py
│   ├── compress_ast_light_pruned.py
│   ├── extract_crossgraphnet_recall.py
│   ├── extract_sid_paths.py
│   ├── filter_dfg_by_split.py
│   ├── filter_dfg_by_split_robust.py
│   ├── fix_ast_ids.py
│   ├── fix_ast_norm_with_cfg.py
│   ├── make_sota_tools_table.py
│   ├── make_tool_vs_crossgraphnet_table.py
│   ├── merge_cfg_contract_level.py
│   ├── merge_sota_tools_matrix.py
│   ├── normalize_ids.py
│   ├── parse_ast_node.js
│   ├── parse_slither_json.py
│   ├── parse_slither_json_tiered.py
│   ├── parse_slither_json_vuln.py
│   ├── rebuild_ast_from_raw.py
│   ├── result_show.py
│   ├── run_sota_oyente.sh
│   ├── run_sota_smartcheck.sh
│   ├── slither_detector_stats.py
│   ├── slither_hit_stats.py
│   ├── sota
│   │   ├── export_sol_by_id.py
│   │   ├── run_sota_tools.sh
│   │   └── summarize_sota_tools.py
│   ├── stat_graph_quailty.py
│   ├── stat_significance.py
│   ├── summarize_all_results.py
│   ├── summarize_dataset_stats.py
│   ├── summarize_full_results.py
│   └── summarize_lite_results.py
├── slither.log
├── solidity.zip
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── data.cpython-310.pyc
│   │   ├── data_dfg.cpython-310.pyc
│   │   ├── data_lite.cpython-310.pyc
│   │   ├── model.cpython-310.pyc
│   │   ├── model_dfg.cpython-310.pyc
│   │   └── train_crosschain.cpython-310.pyc
│   ├── data.py
│   ├── data_dfg.py
│   ├── data_lite.py
│   ├── eval.py
│   ├── federated
│   │   ├── __init__.py
│   │   ├── adapters.py
│   │   ├── client.py
│   │   ├── fedavg.py
│   │   ├── server.py
│   │   └── train_federated.py
│   ├── model.py
│   ├── model_dfg.py
│   ├── run_fl_exps.sh
│   ├── train.py
│   ├── train_crosschain.py
│   └── train_crosschain_lite.py
├── src_backup_20251217_1450
│   ├── collectors
│   │   ├── build_spc_from_datasets.py
│   │   ├── enhanced_spc_builder.py
│   │   ├── etherscan_crawler.py
│   │   ├── etherscan_crawler_5.py
│   │   └── github_spc_crawler.py
│   ├── crossgraphnet
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── dataset.py
│   │   ├── models
│   │   ├── utils
│   │   └── vocab.py
│   ├── dataset
│   │   └── classify_unknown_types.py
│   ├── eval
│   │   ├── analyze_bootstrap_data.py
│   │   ├── filter_best_spc_pairs.py
│   │   ├── merge_all_spc.py
│   │   └── verify_graph_data.py
│   ├── experiments
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── backup
│   │   ├── data_lite.py
│   │   ├── eval_crosschain_lite.py
│   │   └── train_crossgraphnet_lite.py
│   ├── preprocessors
│   │   ├── build_main_dataset_graphs.py
│   │   ├── parallel_parse_worker.py
│   │   ├── simple_graph_builder.py
│   │   ├── split_dataset.py
│   │   └── split_dataset_fixed.py
│   └── utils
│       ├── __init__.py
│       └── data_utils.py
├── test.py
├── text
├── tools
│   ├── make_polygon_proto_main_table.py
│   ├── plot_polygon_f1_threshold.py
│   ├── polygon_oracle_f1_analysis.py
│   ├── summarize_polygon_evalplus.py
│   └── summarize_proto_ablation.py
└── tree-sitter-solidity