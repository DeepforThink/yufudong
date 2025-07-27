import type { Props } from "astro";
import IconMail from "@/assets/icons/IconMail.svg";
import IconGitHub from "@/assets/icons/IconGitHub.svg";
import IconBrandX from "@/assets/icons/IconBrandX.svg";
import IconWechat from "@/assets/icons/IconWechat.svg";
import IconXiaohongshu from "@/assets/icons/IconXiaohongshu.svg";
import IconWhatsapp from "@/assets/icons/IconWhatsapp.svg";
import IconFacebook from "@/assets/icons/IconFacebook.svg";
import IconTelegram from "@/assets/icons/IconTelegram.svg";
import IconPinterest from "@/assets/icons/IconPinterest.svg";
import IconZhihu from "@/assets/icons/IconZhihu.svg";
import { SITE } from "@/config";

interface Social {
  name: string;
  href: string
  linkTitle: string;
  icon: (_props: Props) => Element;
}

export const SOCIALS: Social[] = [
  {
    name: "GitHub",
    href: "https://github.com/deepforthink",
    linkTitle: `${SITE.title} on GitHub`,
    icon: IconGitHub,
  },
  {
    name: "X",
    href: "https://x.com/@PiersonD015",
    linkTitle: `${SITE.title} on X`,
    icon: IconBrandX,
  },
  {
    name: "Xiaohongshu",
    href: "https://www.xiaohongshu.com/user/profile/64903e05000000000f00708b",
    linkTitle: `${SITE.title} on Xiaohongshu`,
    icon: IconXiaohongshu,
  },
  {
    name: "Mail",
    href: "mailto:yf_dong@mail.nankai.edu.cn",
    linkTitle: `Send an email to ${SITE.title}`,
    icon: IconMail,
  },
  {
    name: "Zhihu",
    href: "https://www.zhihu.com/people/liu-lang-de-si-xiang",
    linkTitle: `${SITE.title} on Zhihu`,
    icon: IconZhihu,
  },
  {
    name: "Wechat",
    href: "https://github.com/deepforthink/yufudong/src/assets/images/Wechatscan.jpg",
    linkTitle: `${SITE.title} on Wechat`,
    icon: IconWechat,
  },
] as const;

export const SHARE_LINKS: Social[] = [
  {
    name: "WhatsApp",
    href: "https://wa.me/?text=",
    linkTitle: `Share this post via WhatsApp`,
    icon: IconWhatsapp,
  },
  {
    name: "Facebook",
    href: "https://www.facebook.com/sharer.php?u=",
    linkTitle: `Share this post on Facebook`,
    icon: IconFacebook,
  },
  {
    name: "X",
    href: "https://x.com/intent/post?url=",
    linkTitle: `Share this post on X`,
    icon: IconBrandX,
  },
  {
    name: "Telegram",
    href: "https://t.me/share/url?url=",
    linkTitle: `Share this post via Telegram`,
    icon: IconTelegram,
  },
  {
    name: "Pinterest",
    href: "https://pinterest.com/pin/create/button/?url=",
    linkTitle: `Share this post on Pinterest`,
    icon: IconPinterest,
  },
  {
    name: "Mail",
    href: "mailto:?subject=See%20this%20post&body=",
    linkTitle: `Share this post via email`,
    icon: IconMail,
  },
] as const;
