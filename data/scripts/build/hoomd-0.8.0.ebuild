# Copyright 1999-2008 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2
# $Header: $

inherit cmake-utils fdo-mime

DESCRIPTION="HOOMD performs general purpose molecular dynamics simulations on
NVIDIA GPUs"
HOMEPAGE="http://www.ameslab.gov/hoomd"
SRC_URI="http://www.ameslab.gov/hoomd/downloads/${P}.tar.bz2"

LICENSE="BSD"
SLOT="0"
KEYWORDS="~amd64 ~x86"
IUSE="cuda static debug"

DEPEND="virtual/python
	>=dev-libs/boost-1.32.0
	>=dev-util/cmake-2.6.0
	cuda? ( >=dev-util/nvidia-cuda-toolkit-2.0 )"
RDEPEND="${DEPEND}"

S="${S}/src"

src_compile() {
	local mycmakeargs="
		$(cmake-utils_use_enable cuda CUDA)
		$(cmake-utils_use_enable static STATIC)
		-DENABLE_DOXYGEN=OFF
		-DHONOR_GENTOO_FLAGS=ON"

	cmake-utils_src_compile
}

pkg_postinst() {
	fdo-mime_desktop_database_update
	fdo-mime_mime_database_update
}

pkg_postrm() {
	fdo-mime_desktop_database_update
	fdo-mime_mime_database_update
}
