from argparse import Namespace

import torch


def build_adamW(model: torch.nn.Module, args: Namespace):
    params = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not _match_name_keywords(n, args.lr_backbone_names) and not _match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if _match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer

def _match_name_keywords(name: str, name_keywords: list[str]):
        for b in name_keywords:
            if b in name:
                return True

        return False
