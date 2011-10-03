default:

# This Makefile presumes that the spec file is located in ./SPECS/$SPECFILE
SPECFILE = hoomd.spec
# should we get a Working Directory from cmake instead?
WD ?= $(shell pwd)

RPMBUILD_ARGS = --define "_topdir $(WD)/rpmbuild"
RPMBUILD_ARGS += --define 'debug_package %{nil}'

# automatically specify settings appropriate to a tagged build
ifneq ($(origin TAGVER), undefined)
RPMBUILD_ARGS += --define "version $(TAGVER)"
RPMBUILD_ARGS += --define "refspec v$(TAGVER)"
RPMBUILD_ARGS += --define "release 0"
endif

# allow direct specification of the version, refspec, and release number
ifneq ($(origin VERSION), undefined)
RPMBUILD_ARGS += --define "version $(VERSION)"
endif
ifneq ($(origin REFSPEC), undefined)
RPMBUILD_ARGS += --define "refspec $(REFSPEC)"
endif
ifneq ($(origin RELEASE), undefined)
RPMBUILD_ARGS += --define "release $(RELEASE)"
endif
ifneq ($(origin QUIET),undefined)
RPMBUILD_ARGS += --quiet
endif

rpm: 
	#prepare build environment
	mkdir -p rpmbuild/{SPECS,SOURCES,RPMS,SRPMS,BUILD}
	cp SPECS/$(SPECFILE) rpmbuild/SPECS/
	#call package builder
	PATH=$$PATH:/usr/local/cuda/bin rpmbuild -ba $(RPMBUILD_ARGS) $(WD)/rpmbuild/SPECS/$(SPECFILE)

clean:
	rm -rf rpmbuild

